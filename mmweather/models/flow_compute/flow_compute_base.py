# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmweather.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer)
from mmweather.models.registry import BACKBONES, FLOWS
from mmweather.utils import get_root_logger


class FlowComputeBaseModule(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out







