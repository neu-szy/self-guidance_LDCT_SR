import torch
import itertools
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_distance_same_slice(lq, gt, **kwargs):
    batch, slice_num, patch_num, c, h, w = lq.shape
    gt_distance_same_slice = 0
    lq_distance_same_slice = 0
    count = 0
    gt_same_slice = gt.view(batch, slice_num * patch_num, c * h * w)
    lq_same_slice = lq.view(batch, slice_num * patch_num, c * h * w)
    for pair in itertools.combinations(range(slice_num * patch_num), 2):
        gt_distance_same_slice += torch.norm(gt_same_slice[:, pair[0], :] - gt_same_slice[:, pair[1], :], dim=-1, keepdim=True)
        lq_distance_same_slice += torch.norm(lq_same_slice[:, pair[0], :] - lq_same_slice[:, pair[1], :], dim=-1, keepdim=True)
        count += 1
    gt_distance_same_slice /= count
    lq_distance_same_slice /= count
    return (gt_distance_same_slice.item() + lq_distance_same_slice.item()) / 2


@METRIC_REGISTRY.register()
def calculate_distance_same_level(lq, gt, **kwargs):
    batch, slice_num, patch_num, c, h, w = lq.shape
    gt_distance_same_level = 0
    lq_distance_same_level = 0
    count = 0
    gt_same_level = torch.mean(gt, dim=2).view(batch, slice_num, -1)
    lq_same_level = torch.mean(lq, dim=2).view(batch, slice_num, -1)
    for pair in itertools.combinations(range(slice_num), 2):
        gt_distance_same_level += torch.norm(gt_same_level[:, pair[0], :] - gt_same_level[:, pair[1], :], dim=-1, keepdim=True)
        lq_distance_same_level += torch.norm(lq_same_level[:, pair[0], :] - lq_same_level[:, pair[1], :], dim=-1, keepdim=True)
        count += 1
    gt_distance_same_level /= count
    lq_distance_same_level /= count
    return (gt_distance_same_level.item() + lq_distance_same_level.item()) / 2


@METRIC_REGISTRY.register()
def calculate_distance_diff_level(lq, gt, **kwargs):
    batch, slice_num, patch_num, c, h, w = lq.shape
    return torch.norm(gt.view((batch, -1))-lq.view((batch, -1)), dim=-1, keepdim=True).item()

