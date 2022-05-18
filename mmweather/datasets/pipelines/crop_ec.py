# Copyright (c) OpenMMLab. All rights reserved.
import math
import random

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from ..registry import PIPELINES
from .utils import random_choose_unknown


def crop_data_by_rang(latitude, longitude, crop_lat, crop_lon):
    latitude_mask = (latitude <= crop_lat[1]) & (latitude >= crop_lat[0])

    longitude_mask = (longitude <= crop_lon[1]) & (longitude >= crop_lon[0])
    crop_w_arg = np.argwhere(longitude_mask)[:, 0]
    crop_h_arg = np.argwhere(latitude_mask)[:, 0]
    crop_w_arg, crop_h_arg = (min(crop_w_arg), max(crop_w_arg) + 1), (min(crop_h_arg), max(crop_h_arg) + 1)
    return crop_h_arg, crop_w_arg


@PIPELINES.register_module()
class ECCrop(object):
    """Crop data to random size and aspect ratio.

    A crop of a random proportion of the original image
    and a random aspect ratio of the original aspect ratio is made.
    The cropped image is finally resized to a given size specified
    by 'crop_size'. Modified keys are the attributes specified in "keys".

    This code is partially adopted from
    torchvision.transforms.RandomResizedCrop:
    [https://pytorch.org/vision/stable/_modules/torchvision/transforms/\
        transforms.html#RandomResizedCrop].

    Args:
        keys (list[str]): The images to be resized and random-cropped.
        crop_size (int | tuple[int]): Target spatial size (h, w).
        scale (tuple[float], optional): Range of the proportion of the original
            image to be cropped. Default: (0.08, 1.0).
        ratio (tuple[float], optional): Range of aspect ratio of the crop.
            Default: (3. / 4., 4. / 3.).
        interpolation (str, optional): Algorithm used for interpolation.
            It can be only either one of the following:
            "nearest" | "bilinear" | "bicubic" | "area" | "lanczos".
            Default: "bilinear".
    """

    def __init__(self,
                 target_crop_key,
                 keys,
                 crop_size,
                 random_crop=True
                 # is_pad_zeros=False
                 ):
        # assert keys, 'Keys should not be empty.'
        # if not mmcv.is_tuple_of(crop_size, int):
        #     raise TypeError(
        #         'Elements of crop_size must be int and crop_size must be'
        #         f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')
        self.target_crop_key = target_crop_key
        self.keys = keys
        self.crop_size = crop_size
        self.random_crop = random_crop
        # self.is_pad_zeros = is_pad_zeros

    def _crop(self, data):
        if not isinstance(data, list):
            data_list = [data]
        else:
            data_list = data

        crop_bbox_list = []
        data_list_ = []

        for item in data_list:
            data_h, data_w = item.shape[-3:-1]
            crop_h, crop_w = self.crop_size
            crop_h = min(data_h, crop_h)
            crop_w = min(data_w, crop_w)
            if self.random_crop:
                x_offset = np.random.randint(0, data_w - crop_w + 1)
                y_offset = np.random.randint(0, data_h - crop_h + 1)
            else:
                x_offset = max(0, (data_w - crop_w)) // 2
                y_offset = max(0, (data_h - crop_h)) // 2

            crop_bbox = [x_offset, y_offset, crop_h, crop_w]
            # item_ = item[..., y_offset:y_offset + crop_h, x_offset:x_offset + crop_w, :]
            crop_bbox_list.append(crop_bbox)
            # data_list_.append(item_)

        if not isinstance(data, list):
            return crop_bbox_list[0]
        return crop_bbox_list

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        latitude_ = results[f'{self.target_crop_key}_latitude']
        longitude_ = results[f'{self.target_crop_key}_longitude']
        data = results[self.target_crop_key]
        crop_bbox = self._crop(data)
        left, top, crop_h, crop_w = crop_bbox
        latitude_select = latitude_[top:top + crop_h]
        longitude_select = longitude_[left:left + crop_w]

        crop_lat = (min(latitude_select), max(latitude_select))
        crop_lon = (min(longitude_select), max(longitude_select))
        results[f'crop_lat'] = crop_lat
        results[f'crop_lon'] = crop_lon

        for key in self.keys:
            latitude = results[f'{key}_latitude']
            longitude = results[f'{key}_longitude']
            data = results[key]
            crop_h_arg_range, crop_w_arg_range = crop_data_by_rang(latitude, longitude, crop_lat, crop_lon)
            results[f'{key}_latitude'] = latitude[crop_h_arg_range[0]:crop_h_arg_range[1]]
            results[f'{key}_longitude'] = longitude[crop_w_arg_range[0]:crop_w_arg_range[1]]
            data = data[..., crop_h_arg_range[0]:crop_h_arg_range[1], crop_w_arg_range[0]:crop_w_arg_range[1], :]
            results[key] = data
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, crop_size={self.crop_size}, '
                     f'interpolation={self.interpolation})')
        return repr_str

