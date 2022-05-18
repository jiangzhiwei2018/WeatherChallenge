# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from .sr_backbone_utils import default_init_weights


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2,
        )
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class UpSampleInterpolate(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel, mode='bilinear'):
        super(UpSampleInterpolate, self).__init__()
        self.upsample_func = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=False)
        self.conv_after_upsample = nn.Conv2d(
            in_channels,
            out_channels,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2)

    def forward(self, x):
        return self.conv_after_upsample(self.upsample_func(x))


class UpSampleMixed(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel, upsample_type,
                 interpolation_mode='bilinear'):
        super(UpSampleMixed, self).__init__()
        upsample_type = upsample_type.lower()
        assert upsample_type in ("interpolate", "pixelshuffle")
        self.upsample_model = UpSampleInterpolate(in_channels, out_channels, scale_factor, upsample_kernel,
                                                  mode=interpolation_mode) if upsample_type == "interpolate" \
            else PixelShufflePack(in_channels, out_channels, scale_factor, upsample_kernel)

    def forward(self, x):
        return self.upsample_model(x)






