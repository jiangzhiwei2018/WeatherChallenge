# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import ModulatedDeformConv2dPack
from mmcv.runner import load_checkpoint
from mmcv.cnn import build_activation_layer
from mmweather.models.common import (PixelShufflePack, ResidualBlockNoBN, flow_warp, make_layer)
from mmweather.models.registry import BACKBONES
from mmweather.utils import get_root_logger
from mmweather.models.builder import build_backbone
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp


class BasicWeatherGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, generator_backbone_cfg=dict(type="UnetGenerator"), **kwargs):
        super(BasicWeatherGenerator, self).__init__()

        generator_backbone_cfg.update(in_channels=in_channels, **kwargs)
        self.model = build_backbone(generator_backbone_cfg)

    def forward(self, inx, **kwargs):
        """

        :param inx: (n, c, h, w)
        :return:
        """
        return self.model(inx, **kwargs)


@BACKBONES.register_module()
class BasicWeather(nn.Module):
    def __init__(self, down_hw=(256, 512),
                 input_keys=("Precip", "Radar", "Wind"), output_keys=("Precip", "Radar", "Wind"),
                 generator_backbone_cfg=dict(type="UnetGenerator"), **kwargs):
        super(BasicWeather, self).__init__()
        self.down_hw = down_hw
        self.resize_hw = nn.Upsample(size=down_hw, mode="bilinear", align_corners=True)
        in_channels = len(input_keys) * 20
        out_channels = len(output_keys) * 20
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.model = BasicWeatherGenerator(in_channels=in_channels, out_channels=out_channels,
                                           generator_backbone_cfg=generator_backbone_cfg, **kwargs)

    def forward(self, inx):
        """
        forward_input_dict
        :return:
        """
        n, t, c, h, w = inx.size()
        # combine_inx =
        combine_inx = inx.flatten(1, 2)
        # h_new, w_new = self.down_hw
        # inx_down = self.resize_hw(inx.flatten(0, 1)).unflatten(0, (n, t))
        # print(combine_inx.size())
        res = self.model(combine_inx)
        return {"output": res.unflatten(1, (t, c))}

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
