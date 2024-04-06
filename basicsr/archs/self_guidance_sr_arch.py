import math
from torch import nn
import torch
from basicsr.archs.swinir_arch import RSTB, PatchEmbed, PatchUnEmbed
from basicsr.utils.registry import ARCH_REGISTRY


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class SRHead(nn.Module):
    def __init__(
            self,
            scale,
            out_channel,
            num_feat
    ):
        super(SRHead, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.PReLU()
        )
        self.upsample = Upsample(scale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, out_channel, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        features = self.upsample(x)
        sr = self.conv_last(features)
        return sr


class SelfGuidanceFusionModule(nn.Module):
    def __init__(self, in_channel=1, num_feat=64):
        super().__init__()
        self.channels = num_feat
        self.conv_coronal = nn.Conv2d(in_channel, num_feat, 3, 1, 1)
        self.conv_sagittal = nn.Conv2d(in_channel, num_feat, 3, 1, 1)
        self.conv_axial = nn.Conv2d(in_channel, num_feat, 3, 1, 1)

    def forward(self, coronal, sagittal, axial, degradation_features=None):
        b, _, h, w = axial.shape
        c = self.channels

        coronal_f = self.conv_coronal(coronal)
        sagittal_f = self.conv_sagittal(sagittal)
        axial_f = self.conv_axial(axial)
        qk = torch.matmul(coronal_f.view(b, c, w*h), sagittal_f.view(b, w*h, c)) / math.sqrt(w*h)
        qk = torch.softmax(qk.view(b, c*c), dim=1).view(b, c, c)
        if degradation_features is None:
            return torch.matmul(qk, axial_f.view(b, c, w * h)).view(b, c, w, h)
        else:
            v = axial_f * degradation_features
            return torch.matmul(qk, v.view(b, c, w * h)).view(b, c, w, h) + degradation_features


@ARCH_REGISTRY.register()
class PriorDegradationEstimator(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_feat=96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, num_feat, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(num_feat, num_feat, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(num_feat, num_feat, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(num_feat, num_feat, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(num_feat, num_feat, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(num_feat, out_channel, kernel_size=5, stride=1, padding=0)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


@ARCH_REGISTRY.register()
class SelfGuidanceSR(nn.Module):
    def __init__(
            self,
            in_channel=1,
            out_channel=1,
            img_size=48,
            patch_size=1,
            num_feat=180,
            depths=(6, 6, 6, 6, 6, 6),
            num_heads=(6, 6, 6, 6, 6, 6),
            window_size=8,
            mlp_ratio=2.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            upscale=2,
            resi_connection='1conv'
    ):
        super().__init__()

        self.sgfm = SelfGuidanceFusionModule(in_channel, num_feat)

        self.conv_first = nn.Conv2d(in_channel, num_feat, 3, 1, 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=norm_layer if patch_norm else None)
        self.patches_resolution = self.patch_embed.patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=norm_layer if patch_norm else None)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        layers = []
        for i_layer in range(len(depths)):
            layers.append(RSTB(
                dim=num_feat,
                input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection
            ))
        self.layers = nn.ModuleList(layers)
        self.sr_head = SRHead(upscale, out_channel, num_feat)

    def forward(self, x, avg_coronal=None, avg_sagittal=None, avg_axial=None, feature_pde=None):
        x_size = (x.shape[2], x.shape[3])
        sgfm = self.sgfm(avg_coronal, avg_sagittal, avg_axial, feature_pde)
        sgfm = self.patch_embed(sgfm)
        x = self.conv_first(x)
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size)
            x += sgfm
        x = self.patch_unembed(x, x_size)
        x = self.sr_head(x)
        return x
