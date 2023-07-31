#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Losses.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/12 16:34   Bot Zhao      1.0         None
"""

# import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    """
    xy = [X, U]
    """
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, cfg, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * cfg.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


def make_one_hot_2d(label, classes):
    """

    :param label: BxCxWxH
    :param classes: CLASSES==max(label)
    :return: BxclassesxWxH one-hot encoded
    """
    one_hot = torch.FloatTensor(label.size()[0], classes, label.size()[2], label.size()[3]).zero_().to(label.device)
    target = one_hot.scatter_(1, label.data, 1)
    return target


def make_one_hot_3d(label, classes):
    one_hot = torch.FloatTensor(label.size()[0], classes, label.size()[2], label.size()[3], label.size()[4]).zero_().to(label.device)
    target = one_hot.scatter_(1, label.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=0):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        target = target.long()

        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()

        if len(target.size()) >= 4:
            target = make_one_hot_3d(target.unsqueeze(dim=1), classes=output.size()[1])
        else:
            target = make_one_hot_2d(target.unsqueeze(dim=1), classes=output.size()[1])



        output = F.softmax(output, dim=1)
        # import pdb
        # pdb.set_trace()
        if self.ignore_index==0:
            output = F.softmax(output, dim=1)
            output = output[:, 1:, ...]
            target = target[:, 1:, ...]
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class DiceLoss_sigmoid(nn.Module):
    def __init__(self, smooth=1., ignore_index=0):
        super(DiceLoss_sigmoid, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        target = target.long()

        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()

        # if len(target.size()) >= 4:
        #     target = make_one_hot_3d(target.unsqueeze(dim=1), classes=output.size()[1])
        # else:
        #     target = make_one_hot_2d(target.unsqueeze(dim=1), classes=output.size()[1])



        # output = F.softmax(output, dim=1)
        # import pdb
        # pdb.set_trace()
        # if self.ignore_index==0:
        #     output = F.softmax(output, dim=1)
        #     output = output[:, 1:, ...]
        #     target = target[:, 1:, ...]
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


def computeQualityMeasures(lP, lT):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.1, labelPred > 0.1)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.1, labelPred > 0.1)
    quality["dice"] = dicecomputer.GetDiceCoefficient()

    return quality


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):

        CE_loss = self.cross_entropy(output, target.long())
        # import pdb
        # pdb.set_trace()
        dice_loss = self.dice(output, target.long())
        return CE_loss + dice_loss


def labels_dice(pred, target, classes):
    dices = []
    for i in range(classes):
        temp1 = np.zeros(target.shape)
        temp2 = np.zeros(pred.shape)
        temp1[target == i] = 1
        temp2[pred == i] = 1
        inter = np.sum(temp1*temp2)
        sums = np.sum(temp1) + np.sum(temp2)
        dices.append((2*(inter+0.001))/(sums+0.001))
        # import pdb
        # pdb.set_trace()
    return dices


class KL_LOSS(nn.Module):
    def __init__(self, smooth=1., ignore_index=0):
        super(KL_LOSS, self).__init__()
    def forward(self, x, recon_x, mu, log_var):
        input = x
        # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recon_x, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        # loss = recons_loss + kld_loss
        # if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
        #     loss = recons_loss + kld_loss
        # elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
        #     self.C_max = self.C_max.to(input.device)
        #     C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
        #     loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        # else:
        #     raise ValueError('Undefined loss type.')
        return recons_loss, kld_loss


def calculate_dice_coefficient(y_true, y_pred, smooth=1):
    """
    计算Dice系数
    :param y_true: 真实标签，形状为(H, W)
    :param y_pred: 预测标签，形状为(H, W)
    :param smooth: 平滑因子，避免除以0
    :return: Dice系数，值在[0,1]之间
    """
    intersection = np.sum(y_true * y_pred)
    dice_coefficient = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    return dice_coefficient

def calculate_iou(y_true, y_pred, smooth=1):
    """
    计算IoU值
    :param y_true: 真实标签，形状为(H, W)
    :param y_pred: 预测标签，形状为(H, W)
    :param smooth: 平滑因子，避免除以0
    :return: IoU值，值在[0,1]之间
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# class DiceLoss(nn.Module):
#     """
#     Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
#     For multi-class segmentation `weight` parameter can be used to assign different weights per class.
#     """
#     def __init__(self, classes=4, weight=None, sigmoid_normalization=True, eval_regions: bool=False):
#         super(DiceLoss, self).__init__()
#
#         self.register_buffer('weight', weight)
#         self.normalization = nn.Sigmoid() if sigmoid_normalization else nn.Softmax(dim=1)
#         self.classes = classes
#         self.eval_regions = eval_regions
#
#     def _flatten(self, tensor: torch.tensor) -> torch.tensor:
#         """
#         Flattens a given tensor such that the channel axis is first.
#         The shapes are transformed as follows:
#            (N, C, D, H, W) -> (C, N * D * H * W)
#         """
#         C = tensor.size(1) # number of channels
#         axis_order = (1, 0) + tuple(range(2, tensor.dim())) # new axis order
#         transposed = tensor.permute(axis_order) # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
#         return transposed.contiguous().view(C, -1) # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
#
#     def _reformat_labels(self, seg_mask):
#         """
#         Input format: (batch_size, channels, D, H, W)
#         :param seg_mask:
#         :return:
#         """
#         wt = torch.stack([ seg_mask[:, 0, ...], torch.sum(seg_mask[:, [1, 2, 3], ...], dim=1)], dim=1)
#         tc = torch.stack([ seg_mask[:, 0, ...], torch.sum(seg_mask[:, [1, 3], ...], dim=1)], dim=1)
#         et = torch.stack([ seg_mask[:, 0, ...], seg_mask[:, 3, ...]], dim=1)
#         return wt, tc, et
#
#     def dice(self, input: torch.tensor, target: torch.tensor, weight: float, epsilon=1e-6) -> float:
#         """
#         Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
#         Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
#         :param input: NxCxSpatial input tensor
#         :param target:  NxCxSpatial target tensor
#         :param weight: Cx1 tensor of weight per channel. Channels represent the class
#         :param epsilon: prevents division by zero
#         :return: dice loss, dice score
#         """
#         assert input.size() == target.size(), "'input' and 'target' must have the same shape"
#
#         input = self._flatten(input)
#         target = self._flatten(target)
#         target = target.float()
#
#         # Compute per channel Dice Coefficient
#         intersect = (input * target).sum(-1)
#         if weight is not None:
#             intersect = weight * intersect
#
#         union = (input * input).sum(-1) + (target * target).sum(-1)
#         return 2 * (intersect / union.clamp(min=epsilon))
#
#
#     def forward(self, input: torch.tensor, target: torch.tensor) -> Tuple[float, float, list]:
#
#         target = utils.expand_as_one_hot(target.long(), self.classes)
#
#         assert input.dim() == target.dim() == 5, f"'input' {input.dim()} and 'target' {target.dim()} have different number of dims "
#
#         input = self.normalization(input.float())
#
#         if self.eval_regions:
#             input_wt, input_tc, input_et = self._reformat_labels(input)
#             target_wt, target_tc, target_et = self._reformat_labels(target)
#
#             wt_dice = torch.mean(self.dice(input_wt, target_wt, weight=self.weight))
#             tc_dice = torch.mean(self.dice(input_tc, target_tc, weight=self.weight))
#             et_dice = torch.mean(self.dice(input_et, target_et, weight=self.weight))
#
#             wt_loss = 1 - wt_dice
#             tc_loss = 1 - tc_dice
#             et_loss = 1 - et_dice
#
#             loss = 1/3 * (wt_loss + tc_loss + et_loss)
#             score = 1/3 * (wt_dice + tc_dice + et_dice)
#
#             return loss, score, [wt_loss, tc_loss, et_loss]
#
#         else:
#             per_channel_dice = self.dice(input, target, weight=self.weight) # compute per channel Dice coefficient
#
#             mean = torch.mean(per_channel_dice)
#             loss = (1. - mean)
#             # average Dice score across all channels/classes
#             return loss, mean, per_channel_dice[1:]