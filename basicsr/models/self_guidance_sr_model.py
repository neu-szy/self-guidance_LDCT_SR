import numpy as np
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import itertools
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.models.map_model import MapModel
from copy import deepcopy


@MODEL_REGISTRY.register()
class SelfGuidanceModel(BaseModel):
    def __init__(self, opt):
        super(SelfGuidanceModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.pde_mode = opt.get('pde_mode', "none")
        if self.pde_mode == "pretrained":
            self.net_pde = build_network(opt['network_pde'])
            self.net_pde = self.model_to_device(self.net_pde)
            self.net_pde.eval()
            self.print_network(self.net_pde)
            pde_load_path = self.opt['path'].get('pretrain_network_degradation_estimator', None)
            if pde_load_path is not None:
                param_key = self.opt['path'].get('param_key_g', 'params')
                self.load_network(self.net_pde, pde_load_path, self.opt['path'].get('strict_load_g', True), param_key)
            else:
                raise ValueError("Pretrained Degradation Estimator weight not found")
        else:
            self.net_pde = None

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g wi
            # th Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if self.cri_pix is None and self.cri_tv is None and self.cri_perceptual is None and self.cri_gan is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'avg_coronal' in data:
            self.avg_coronal = data['avg_coronal'].to(self.device)
        else:
            self.avg_coronal = self.lq
        if 'avg_sagittal' in data:
            self.avg_sagittal = data['avg_sagittal'].to(self.device)
        else:
            self.avg_sagittal = self.lq
        if 'avg_axial' in data:
            self.avg_axial = data['avg_axial'].to(self.device)
        else:
            self.avg_axial = self.lq

    def optimize_parameters(self, current_iter):
        if self.pde_mode == "pretrained":
            with torch.no_grad():
                self.pde_feature = self.net_pde(self.lq)
        elif self.pde_mode == "none":
            self.pde_feature = None
        else:
            raise KeyError(f"unsupported pde mode: {self.pde_mode}")

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.avg_coronal, self.avg_sagittal, self.avg_axial, self.pde_feature)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_tv:
            l_tv = self.cri_tv(self.output, self.gt)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv
        if self.cri_gan:
            l_gan_pred = self.cri_gan(self.output, False)
            l_gan_gt = self.cri_gan(self.gt, True)
            l_total += l_gan_pred
            l_total += l_gan_gt
            loss_dict['l_gan'] = l_gan_pred + l_gan_gt

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        return l_total.item()

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.pde_mode == "pretrained":
                    self.pde_feature = self.net_pde(self.lq)

                elif self.pde_mode == "none":
                    self.pde_feature = None
                else:
                    raise KeyError(f"unsupported pde mode: {self.pde_mode}")
                self.output = self.net_g_ema(self.lq, self.avg_coronal, self.avg_sagittal, self.avg_axial, self.pde_feature)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.pde_mode == "pretrained":
                    self.pde_feature = self.net_pde(self.lq)
                elif self.pde_mode == "none":
                    self.pde_feature = None
                else:
                    raise KeyError(f"unspport pde mode: {self.pde_mode}")
                self.output = self.net_g(self.lq, self.avg_coronal, self.avg_sagittal, self.avg_axial, self.pde_feature)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        import numpy as np
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(deepcopy(metric_data), opt_)

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                if self.output_flag == "float32":
                    sr_img = np.uint8(sr_img * 255)
                imwrite(sr_img, save_img_path)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            # TB.add_scalar(metric, value, current_iter)
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class PriorDegradationEstimatorModel(MapModel):
    def __init__(self, opt):
        super().__init__(opt)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('distance_same_slice_opt'):
            self.cri_distance_same_slice = build_loss(train_opt['distance_same_slice_opt']).to(self.device)
        else:
            self.cri_distance_same_slice = None
        if train_opt.get('distance_same_level_opt'):
            self.cri_distance_same_level = build_loss(train_opt['distance_same_level_opt']).to(self.device)
        else:
            self.cri_distance_same_level = None
        if train_opt.get('distance_diff_level_opt'):
            self.cri_distance_diff_level = build_loss(train_opt['distance_diff_level_opt']).to(self.device)
        else:
            self.cri_distance_diff_level = None
        if train_opt.get('rank_opt'):
            self.cri_rank = build_loss(train_opt['rank_opt']).to(self.device)
        else:
            self.cri_rank = None

        if self.cri_distance_same_slice is None and self.cri_distance_same_level is None and self.cri_distance_diff_level is None and self.cri_rank is None:
            raise ValueError('Both losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        # shape: batch, slice, patch_num, channel, height, weight
        if 'gt_patch_lists' in data:
            self.gt_patch_lists = data['gt_patch_lists'].to(self.device)
        else:
            raise KeyError("gt_patch_lists not found")
        if 'lq_patch_lists' in data:
            self.lq_patch_lists = data['lq_patch_lists'].to(self.device)
        else:
            raise KeyError("lq_patch_lists not found")

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.output_gt_patch = torch.zeros_like(self.gt_patch_lists).to(self.gt_patch_lists.device)
        self.output_lq_patch = torch.zeros_like(self.lq_patch_lists).to(self.lq_patch_lists.device)

        batch, slice_num, patch_num, c, h, w = self.gt_patch_lists.shape
        for slice_ind in range(slice_num):
            for patch_ind in range(patch_num):
                self.output_gt_patch[:, slice_ind, patch_ind, :, :, :] = self.net_g(self.gt_patch_lists[:, slice_ind, patch_ind, :, :, :])
                self.output_lq_patch[:, slice_ind, patch_ind, :, :, :] = self.net_g(self.lq_patch_lists[:, slice_ind, patch_ind, :, :, :])

        # 计算一个slice中不同patch间的平均距离
        gt_distance_of_same_slice = 0
        lq_distance_of_same_slice = 0
        count = 0
        output_gt_patch = self.output_gt_patch.view(batch, slice_num * patch_num, c * h * w)
        output_lq_patch = self.output_lq_patch.view(batch, slice_num * patch_num, c * h * w)

        for pair in itertools.combinations(range(slice_num * patch_num), 2):
            gt_distance_of_same_slice += torch.norm(output_gt_patch[:, pair[0], :] - output_gt_patch[:, pair[1], :], dim=-1, keepdim=True)
            lq_distance_of_same_slice += torch.norm(output_lq_patch[:, pair[0], :] - output_lq_patch[:, pair[1], :], dim=-1, keepdim=True)
            count += 1
        gt_distance_of_same_slice /= count
        lq_distance_of_same_slice /= count

        # 计算一个level不同slice间的平均距离
        gt_distance_of_same_level = 0
        lq_distance_of_same_level = 0
        count = 0

        output_gt_slice = torch.mean(self.output_gt_patch, dim=2).view(batch, slice_num, -1)
        output_lq_slice = torch.mean(self.output_lq_patch, dim=2).view(batch, slice_num, -1)

        for pair in itertools.combinations(range(slice_num), 2):
            gt_distance_of_same_level += torch.norm(output_gt_slice[:, pair[0], :] - output_gt_slice[:, pair[1], :], dim=-1, keepdim=True)
            lq_distance_of_same_level += torch.norm(output_lq_slice[:, pair[0], :] - output_lq_slice[:, pair[1], :], dim=-1, keepdim=True)
            count += 1

        gt_distance_of_same_level /= count
        lq_distance_of_same_level /= count

        # 计算不同level间的平均距离
        output_gt_level = self.output_gt_patch.view(batch, -1)
        output_lq_level = self.output_lq_patch.view(batch, -1)

        distance = torch.norm(output_gt_level - output_lq_level, dim=-1, keepdim=True)

        l_total = 0
        loss_dict = OrderedDict()
        # distance_loss
        if self.cri_distance_same_slice:
            l_distance_same_slice = self.cri_distance_same_slice(lq_distance_of_same_slice,
                                                                 torch.zeros_like(lq_distance_of_same_slice).to(lq_distance_of_same_slice.device)) \
                                    + self.cri_distance_same_slice(gt_distance_of_same_slice,
                                                                   torch.zeros_like(gt_distance_of_same_slice).to(lq_distance_of_same_slice.device))
            l_total += l_distance_same_slice
            loss_dict['l_distance_same_slice'] = l_distance_same_slice

        if self.cri_distance_same_level:
            l_distance_same_level = - self.cri_distance_same_level(gt_distance_of_same_level,
                                                                   torch.zeros_like(gt_distance_of_same_level).to(gt_distance_of_same_level.device)) \
                                    - self.cri_distance_same_level(lq_distance_of_same_level,
                                                                   torch.zeros_like(lq_distance_of_same_level).to(lq_distance_of_same_level.device))
            l_total += l_distance_same_level
            loss_dict['l_distance_same_level'] = -l_distance_same_level

        if self.cri_distance_diff_level:
            l_distance_diff_level = - self.cri_distance_diff_level(distance, torch.zeros_like(distance).to(distance.device))
            l_total += l_distance_diff_level
            loss_dict['l_distance_diff_level'] = -l_distance_diff_level

        # rank_loss
        if self.cri_rank:
            # Assumption a: noise features of same slice are more similar than different slices
            l_rank_a_lq = self.cri_rank(lq_distance_of_same_slice, lq_distance_of_same_level,
                                        -torch.zeros_like(lq_distance_of_same_slice).to(lq_distance_of_same_slice.device))
            l_rank_a_gt = self.cri_rank(gt_distance_of_same_slice, gt_distance_of_same_level,
                                        -torch.zeros_like(gt_distance_of_same_slice).to(gt_distance_of_same_slice.device))
            l_total += l_rank_a_lq
            l_total += l_rank_a_gt
            loss_dict['l_rank_a_lq'] = l_rank_a_lq
            loss_dict['l_rank_a_gt'] = l_rank_a_gt

            # Assumption b: noise features of same level are more similar than different levels
            l_rank_b = self.cri_rank(lq_distance_of_same_level + gt_distance_of_same_level, 2 * distance,
                                     -torch.zeros_like(lq_distance_of_same_level).to(lq_distance_of_same_level.device))
            l_total += l_rank_b
            loss_dict['l_rank_b'] = l_rank_b

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.output_gt_patch = torch.zeros_like(self.gt_patch_lists).to(self.gt_patch_lists.device)
        self.output_lq_patch = torch.zeros_like(self.lq_patch_lists).to(self.lq_patch_lists.device)
        batch, slice_num, patch_num, c, h, w = self.gt_patch_lists.shape
        self.net_g.eval()
        with torch.no_grad():
            for slice_ind in range(slice_num):
                for patch_ind in range(patch_num):
                    self.output_gt_patch[:, slice_ind, patch_ind, :, :, :] = self.net_g(self.gt_patch_lists[:, slice_ind, patch_ind, :, :, :])
                    self.output_lq_patch[:, slice_ind, patch_ind, :, :, :] = self.net_g(self.lq_patch_lists[:, slice_ind, patch_ind, :, :, :])
        self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            metric_data['lq'] = self.output_lq_patch.cpu()
            metric_data['gt'] = self.output_gt_patch.cpu()
            del self.output_lq_patch
            del self.output_gt_patch

            # tentative for out of GPU memory
            del self.gt_patch_lists
            del self.lq_patch_lists
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
