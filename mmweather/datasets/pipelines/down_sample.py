# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
from mmcv import imresize
from ..registry import PIPELINES
import torch.nn.functional as F


@PIPELINES.register_module()
class MyDownSampling:
    def __init__(self, keys, save_original=True, **inter_kwargs):
        """

        :param args:
        :param kwargs:
        """
        self.keys = keys
        self.inter_kwargs = inter_kwargs
        self.save_original = save_original

    def __call__(self, results):
        scale_factor = 1 / results["hw_scale"]
        for key in self.keys:
            data = results[key]
            results[key] = F.interpolate(input=data, scale_factor=scale_factor,
                                         align_corners=True,
                                         **self.inter_kwargs)
        return results


@PIPELINES.register_module()
class MyDownSamplingFrames:
    def __init__(self, keys, **inter_kwargs):
        """

        :param args:
        :param kwargs:
        """
        self.keys = keys
        self.inter_kwargs = inter_kwargs
        # self.save_original = save_original

    def __call__(self, results):
        for key in self.keys:
            data = results[key]
            results[key] = data[[0, -1]]
        return results
