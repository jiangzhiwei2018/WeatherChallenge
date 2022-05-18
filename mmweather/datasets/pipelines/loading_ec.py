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


class __ECDataIter(Dataset):
    def __init__(self, ec_data_file):
        self.len = ec_data_file.shape[0]
        self.data = ec_data_file

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.len


@PIPELINES.register_module()
class LoadECdataFrames:
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

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        data_type_names = results[f'data_types']
        frameIndex = results["frameIndex"]
        # print(results['start_frame'])
        for key in self.keys:
            start_frame_idx = results['start_frame']
            end_frame_idx = results['end_frame']
            ec_data_file = results[f'{key}_ec_data_file_is_read']
            crop_h_arg = results[f'{key}_crop_h_arg']
            crop_w_arg = results[f'{key}_crop_w_arg']
            concat_data = []
            for data_name in data_type_names:
                ec_data = ec_data_file[data_name]
                n_frame = np.array(ec_data[
                                   start_frame_idx:end_frame_idx + 1,
                                   ...,
                                   crop_h_arg[0]:crop_h_arg[1], crop_w_arg[0]:crop_w_arg[1]],
                                   copy=True,
                                   dtype=np.float32)
                mid_frame = n_frame[frameIndex]
                n_frame = np.concatenate((n_frame[0:1], mid_frame, n_frame[-1:]))
                concat_data.append(n_frame[..., None])
            concat_data = np.concatenate(concat_data)
            results[key] = concat_data.copy()
            results["frameIndex"] = frameIndex.astype(np.float32)
        return results

    # def __repr__(self):
    #     repr_str = self.__class__.__name__
    #     repr_str += (
    #         f'(io_backend={self.io_backend}, key={self.key}, '
    #         f'flag={self.flag}, save_original_img={self.save_original_img}, '
    #         f'channel_order={self.channel_order}, use_cache={self.use_cache})')
    #     return repr_str
