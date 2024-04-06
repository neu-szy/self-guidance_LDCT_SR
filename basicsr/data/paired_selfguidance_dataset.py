import os

import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import cv2
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, augment_following_status
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
import numpy as np
import random
from PIL import Image


def create_avg_ct_dic(folder):
    avg_ct_dic = {}
    avg_names = os.listdir(folder)
    for avg_name in avg_names:
        num = os.path.splitext(avg_name)[0]
        avg_ct_dic[num] = os.path.join(folder, avg_name)
    return avg_ct_dic


@DATASET_REGISTRY.register()
class PairedSelfGuidanceDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # disk, lmdb ,...
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.scale = self.opt['scale']


        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.coronal_folder = opt['dataroot_avg_coronal']
        self.sagittal_folder = opt['dataroot_avg_sagittal']
        self.axial_folder = opt['dataroot_avg_axial']

        self.avg_coronal_dic = create_avg_ct_dic(self.coronal_folder)
        self.avg_sagittal_dic = create_avg_ct_dic(self.sagittal_folder)
        self.avg_axial_dic = create_avg_ct_dic(self.axial_folder)

        self.rgb = opt.get("rgb", False)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.opt['meta_info_file'],
                                                          self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)


        gt_path = self.paths[index]['gt_path']
        img_gt = load_img(gt_path)
        img_gt = np.expand_dims(img_gt, 2)
        lq_path = self.paths[index]['lq_path']
        img_lq = load_img(lq_path)
        img_lq = np.expand_dims(img_lq, 2)
        ct_num1 = os.path.basename(lq_path)
        ct_num1 = os.path.splitext(ct_num1)[0]
        ct_num1 = ct_num1.split("_")[0]

        avg_coronal_path = self.avg_coronal_dic[ct_num1]
        avg_sagittal_path = self.avg_sagittal_dic[ct_num1]
        avg_axial_path = self.avg_axial_dic[ct_num1]

        img_avg_coronal = load_img(avg_coronal_path)
        img_avg_sagittal = load_img(avg_sagittal_path)
        img_avg_axial = load_img(avg_axial_path)
        img_avg_coronal = np.expand_dims(img_avg_coronal, 2)
        img_avg_sagittal = np.expand_dims(img_avg_sagittal, 2)
        img_avg_axial = np.expand_dims(img_avg_axial, 2)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.scale, gt_path)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            # img_gt_lr = rgb2ycbcr(img_gt_lr, y_only=True)[..., None]
            # img_mask = rgb2ycbcr(img_mask, y_only=True)[..., None]
            img_avg_coronal = rgb2ycbcr(img_avg_coronal, y_only=True)[..., None]
            img_avg_sagittal = rgb2ycbcr(img_avg_sagittal, y_only=True)[..., None]
            img_avg_axial = rgb2ycbcr(img_avg_axial, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * self.scale, 0:img_lq.shape[1] * self.scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_avg_coronal, img_avg_sagittal, img_avg_axial = img2tensor(
            [img_gt, img_lq, img_avg_coronal, img_avg_sagittal, img_avg_axial],
            bgr2rgb=True, float32=True)
        img_avg_coronal = F.interpolate(img_avg_coronal.unsqueeze(0), img_lq.shape[1:], mode='bicubic').squeeze(0)
        img_avg_sagittal = F.interpolate(img_avg_sagittal.unsqueeze(0), img_lq.shape[1:], mode='bicubic').squeeze(0)
        img_avg_axial = F.interpolate(img_avg_axial.unsqueeze(0), img_lq.shape[1:], mode='bicubic').squeeze(0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_avg_coronal, self.mean, self.std, inplace=True)
            normalize(img_avg_sagittal, self.mean, self.std, inplace=True)
            normalize(img_avg_axial, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'avg_coronal': img_avg_coronal,
            'avg_sagittal': img_avg_sagittal,
            'avg_axial': img_avg_axial,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'avg_coronal_path': avg_coronal_path,
            'avg_sagittal_path': avg_sagittal_path,
            'avg_axial_path': avg_axial_path,
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PrioriDegradationEstimatorDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # disk, lmdb ,...
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.slice_num = opt.get('slice_num', 1)
        self.patch_num = opt.get('patch_num', 1)

        self.scale = opt.get("scale", 1)

        self.rgb = opt.get("rgb", False)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.opt['meta_info_file'],
                                                          self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_patch_lists = []
        lq_patch_lists = []

        # generate random indexes
        index_list = [index]
        while len(index_list) < self.slice_num:
            index_tmp = random.randint(0, len(self.paths) - 1)
            while index_tmp in index_list:
                index_tmp = random.randint(0, len(self.paths) - 1)
            index_list.append(index_tmp)
        for ind in index_list:
            gt_patches = []
            lq_patches = []

            # Load gt and lq images. Dimension order: HWC; channel order: BGR;
            # image range: [0, 1], float32.
            gt_path = self.paths[ind]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True, flag='grayscale')
            img_gt = np.expand_dims(img_gt, 2)

            lq_path = self.paths[ind]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True, flag='grayscale')
            img_lq = np.expand_dims(img_lq, 2)

            gt_size = self.opt['gt_size']
            if self.opt['use_flip']:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
            else:
                hflip = False
                vflip = False
            if self.opt['use_rot']:
                rot90 = random.random() < 0.5
            else:
                rot90 = False
            for _ in range(self.patch_num):
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.scale, gt_path)
                # flip, rotation
                img_gt, img_lq = augment_following_status([img_gt, img_lq], hflip=hflip, vflip=vflip, rot90=rot90)
                gt_patches.append(img_gt)
                lq_patches.append(img_lq)

            if 'color' in self.opt and self.opt['color'] == 'y':
                gt_patches = [rgb2ycbcr(gt, y_only=True)[..., None] for gt in gt_patches]
                lq_patches = [rgb2ycbcr(lq, y_only=True)[..., None] for lq in lq_patches]

            # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
            if self.opt['phase'] != 'train':
                gt_patches = [gt[0:lq_patches[0].shape[0] * self.scale, 0:lq_patches[0].shape[1] * self.scale, :] for gt in gt_patches]

            # BGR to RGB, HWC to CHW, numpy to tensor
            gt_patches = img2tensor(gt_patches, bgr2rgb=False, float32=True)
            lq_patches = img2tensor(lq_patches, bgr2rgb=False, float32=True)

            # normalize
            if self.mean is not None or self.std is not None:
                for gt in gt_patches:
                    normalize(gt, self.mean, self.std, inplace=True)
                for lq in lq_patches:
                    normalize(lq, self.mean, self.std, inplace=True)
            gt_patch_lists.append(torch.stack(gt_patches, dim=0))
            lq_patch_lists.append(torch.stack(gt_patches, dim=0))

        return {
            'gt_patch_lists': torch.stack(gt_patch_lists, dim=0),
            'lq_patch_lists': torch.stack(lq_patch_lists, dim=0)
        }

    def __len__(self):
        return len(self.paths)


def load_img(img_path):
    _, ext = os.path.split(img_path)
    if ext == ".npy":
        img = np.load(img_path, allow_pickle=True).astype(np.float32)
    else:
        img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    return img