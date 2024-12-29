# Copyright 2022 CircuitNet. All rights reserved.

import functools

import torch.nn as nn
import torch.nn.functional as F

import utils.losses as losses
import torch

# SSIM算法见 https://blog.csdn.net/qq_35914625/article/details/113789903

from torch.autograd import Variable
import numpy as np
from math import exp


def build_loss(opt):
    return losses.__dict__[opt.pop("loss_type")]()


__all__ = ["L1Loss", "MSELoss", "NRMSELoss", "SSIMLoss", "MSEandSSIM", "NRMSEandSSIM"]


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()

    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction="mean", sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == "sum":
        loss = reduce_loss(loss, reduction)
    elif reduction == "mean":
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)  # 将权重扩展为与损失张量 loss 的形状相同
        eps = 1e-12

        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss


def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(
        pred, target, weight=None, reduction="mean", sample_wise=False, **kwargs
    ):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper


@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction="mean", sample_wise=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise
        )


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(
        pred, target, reduction="none"
    )  # 这里恒为 none 因为传参并没有传reduction


class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction="mean", sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise
        )


def nrmse_loss(pred, target):
    mse_loss = F.mse_loss(pred, target, reduction="none")
    mse = mse_loss.mean()
    rmse = torch.sqrt(mse)
    target_range = target.max() - target.min()
    nrmse = rmse / (target_range + 1e-12)

    return nrmse


class NRMSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction="mean", sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * nrmse_loss(pred, target)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()  # window_size


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # window_size * 1
    _2D_window = (
        _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    )  # 1*1*window_size*window_size
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )

    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=100.0, window_size=11, size_average=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.ssim = SSIM(window_size, size_average)

    def forward(self, pred, target, weight=None):
        return self.loss_weight * (1 - self.ssim(pred, target))


class MSEandSSIM(nn.Module):
    def __init__(
        self,
        loss_weight=100.0,
        reduction="mean",
        sample_wise=False,
        window_size=11,
        size_average=True,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.ssim = SSIM(window_size, size_average)

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise
        ) + (1 - self.ssim(pred, target))


class NRMSEandSSIM(nn.Module):
    def __init__(
        self,
        loss_weight=100.0,
        reduction="mean",
        sample_wise=False,
        window_size=11,
        size_average=True,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.ssim = SSIM(window_size, size_average)

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * nrmse_loss(pred, target) + (
            1 - self.ssim(pred, target)
        )
