# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
import numpy as np
# import mmcv
# import numpy as np
# from mmcv.fileio import FileClient
#
# from mmweather.core.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask,
#                               random_bbox)
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset, IterableDataset, Subset
import os
from ..registry import PIPELINES
import random
import json
from imageio import imread


def img_read(frame, factor):
    """

    :param frame:
    :param factor:
    :return:
    """
    image = np.array(imread(frame), dtype=np.float32)[None, None]/255*factor
    return image


def img_read_list(frame_list, factor=1.):
    # if isinstance(frame_list, np.ndarray):
    #     return frame_list.astype(np.float32)/255.*factor
    img_list = []
    for frame in frame_list:
        img_list.append(img_read(frame, factor))
    img_list = np.concatenate(img_list, axis=0)
    return img_list


@PIPELINES.register_module()
class LoadImages:
    """Load image from file.

    """

    def __init__(self,
                 keys,
                 *args,
                 **kwargs):
        """

        :param args:
        :param kwargs:
        """
        self.keys = keys
        self.data_type_factor = {"Precip": 10, "Radar": 70, "Wind": 35}

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        input_img = []
        need_out = None
        for key in self.keys:
            input_pth_list, _ = results[key]
            input_img_list = img_read_list(input_pth_list, factor=1.)
            input_img.append(input_img_list)
            # results[key] = res_
            need_out = _
        results['input_img'] = np.concatenate(input_img, axis=1)
        if need_out is not None:
            gt_res = []
            for key in self.keys:
                _, output_pth_list = results[key]
                out_img_list = img_read_list(output_pth_list, factor=1.)
                gt_res.append(out_img_list)
            results['gt'] = np.concatenate(gt_res, axis=1)
        return results
