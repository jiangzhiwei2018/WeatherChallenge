# Copyright (c) OpenMMLab. All rights reserved.
import math
import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from math import exp


def _psnr(img1, img2, max_val=50., *args, **kwargs):
    if max_val is None:
        max_val = img2.max()
    mse_value = float(((img1 - img2) ** 2).mean())
    if mse_value == 0:
        return float('inf')
    return 20. * math.log10(max_val / math.sqrt(mse_value))


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def torchssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None,
              *args, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        L = img2.max() - img2.min()
        # if torch.max(img1) > 128:
        #     max_val = 255
        # else:
        #     max_val = 1
        # if torch.min(img1) < -0.5:
        #     min_val = -1
        # else:
        #     min_val = 0
        # L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (channel, height, width) = img1.size(-3), img1.size(-2), img1.size(-1)
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)  # .to(img1.device)
    # ipdb.set_trace()
    window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)  # 高斯滤波 求均值
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)  # 求均值

    mu1_sq = mu1.pow(2)  # 平方
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq  # var(x) = Var(X)=E[X^2]-E[X]^2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2  # 协方差

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    return float(ssim_map.mean())
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM:
    def __init__(self, window_size=11, size_average=True, val_range=None, channel=1, channel_dim=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = channel
        self.channel_dim = channel_dim
        self.window = create_window(window_size)

    def __call__(self, img1, img2, *args, **kwargs):
        channel = img1.size(self.channel_dim)

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        # for n_channel_id in channel:
        return torchssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average,
                         val_range=self.val_range, **kwargs)


# Classes to re-use window
class PSNR:
    def __init__(self, max_val=50.):
        super(PSNR, self).__init__()
        self.max_val = max_val

    def __call__(self, img1, img2, *args, **kwargs):
        return _psnr(img1, img2, self.max_val)



