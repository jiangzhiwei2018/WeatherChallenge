# Copyright (c) OpenMMLab. All rights reserved.
import random
import pickle
import mmcv
import numpy as np
import os
from ..registry import PIPELINES
import json
import netCDF4 as nc
import torch


def merge_mean_std(n, mean1, std1, m, mean2, std2):
    """
    Given the number, mean and standard deviation of the two groups of data, \
    calculate the mean and standard deviation of the total data
    :param m:
    :param n: number of data in the first group
    :param mean1: mean value of the first set of data
    :param std1: standard deviation of the first set of data
    :param: number of data in the second group
    :param mean2: mean value of the second group of data
    :param std2: standard deviation of the second set of data
    :return: the number, mean and standard deviation of all data
    """
    mean = (n * mean1 + m * mean2) / (m + n)
    std = np.sqrt((n * (std1 ** 2 + mean1 ** 2) + m * (std2 ** 2 + mean2 ** 2)) / (m + n) - mean ** 2)
    return m + n, mean, std


def get_mean_std(arr, axis=None, keepdims=False, **kwargs):
    mean = np.mean(arr, axis=axis, keepdims=keepdims)
    std = np.std(arr, axis=axis, keepdims=keepdims, ddof=1)
    return mean, std


def get_data_batch_one(inputs, batch_size=None, shuffle=False,
                       crop_w_arg=None,
                       crop_h_arg=None, **kwargs
                       ):
    '''
    产生批量数据batch,非循环迭代
    迭代次数由:iter_nums= math.ceil(sample_nums / batch_size)
    :param crop_h_arg:
    :param crop_w_arg:
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    # rows,cols=inputs.shape
    rows = len(inputs)
    indices = list(range(rows))
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        cur_nums = len(indices)
        batch_size = np.where(cur_nums > batch_size, batch_size, cur_nums)
        batch_indices = indices[0:batch_size]  # 产生一个batch的index
        indices = indices[batch_size:]
        # indices = indices[batch_size:] + indices[:batch_size]  # 循环移位，以便产生下一个batch
        batch_data = inputs[batch_indices]
        if crop_h_arg is not None:
            # print(crop_h_arg)
            batch_data = batch_data[..., crop_h_arg, :]
        if crop_w_arg is not None:
            batch_data = batch_data[..., crop_w_arg]
        yield batch_data


def get_all_data_mean_std(data_nc_file, batch_size=256, max_it=100,
                          shuffle=False,
                          **kwargs):
    """

    :param batch_size:
    :param shuffle:
    :param max_it:
    :param data_nc_file:
    :param batch_size:
    :return:
    """
    # data_ld = DataLoader(__ECDataIter(data_nc_file), batch_size=bs, shuffle=False)
    data_ld = get_data_batch_one(data_nc_file, shuffle=shuffle, batch_size=batch_size, **kwargs)
    last_data_num = None
    last_data_std = None
    last_data_mean = None
    all_size = min(int(np.ceil(len(data_nc_file) / batch_size)), max_it)
    # print(all_size)
    for idx in range(all_size):
        n_data = data_ld.__next__()
        # n_data = n_data.cpu().numpy()

        mean, std = get_mean_std(n_data, **kwargs)
        if idx > 0:
            last_data_num, mean, std = merge_mean_std(last_data_num, last_data_mean, last_data_std,
                                                      len(n_data), mean, std)
        else:
            last_data_num = len(n_data)
        last_data_std = std
        last_data_mean = mean
        print(n_data.shape)
        print(last_data_mean)
    return last_data_mean, last_data_std


# def normal(data_in, data_mean, data_std):
#     """
#
#     :return:
#     """
#     mmcv.imdenormalize()


# @PIPELINES.register_module()
def float2array(val):
    if isinstance(val, (float, np.float_, torch.FloatType)):
        val = np.array([val])
    return val


class GetMeanStd:
    def __init__(self, use_cache=True, mean_std_save_cache_folder=None,
                 crop_area_lon=None,
                 crop_area_lat=None,
                 latitude_name="latitude",
                 longitude_name="longitude",
                 # batch_size=256, axis=(0, ), keepdims=False, max_it=100, shuffle=False,
                 **kwargs):
        self.use_cache = use_cache
        if not mean_std_save_cache_folder:
            mean_std_save_cache_folder = os.path.join(".", "default_cache")
        self.mean_std_save_cache_folder = mean_std_save_cache_folder
        os.makedirs(self.mean_std_save_cache_folder, exist_ok=True)
        self.mean_std_json_pth = os.path.join(self.mean_std_save_cache_folder, "data_mean_std_dict.pkl")
        if os.path.exists(self.mean_std_json_pth):
            with open(self.mean_std_json_pth, 'rb') as fr:
                self.mean_std = pickle.load(fr)
            # json.load(self.mean_std)
            # print(json.loads(str(self.mean_std)))
        else:
            self.mean_std = dict()
        self.kwargs = kwargs
        self.crop_lon = crop_area_lon
        self.crop_lat = crop_area_lat
        self.latitude_name = latitude_name
        self.longitude_name = longitude_name
        # self.crop_h_arg, self.crop_w_arg = None, None

    def __call__(self, ec_data_file, data_type_names):
        """Call function.

        Args:


        Returns:
            dict: A dict containing the processed data and information.
        """
        if isinstance(ec_data_file, str):
            ec_data_file = nc.Dataset(ec_data_file, 'r')
        mean_std = dict()
        crop_w_arg = None
        crop_h_arg = None
        if self.crop_lat:
            latitude = ec_data_file[self.latitude_name][:]
            latitude_mask = (latitude <= self.crop_lat[1]) & (latitude >= self.crop_lat[0])
            crop_h_arg = latitude_mask
            # print(f"crop_lat by {self.crop_lat}")
        if self.crop_lon:
            longitude = ec_data_file[self.longitude_name][:]
            longitude_mask = (longitude <= self.crop_lon[1]) & (longitude >= self.crop_lon[0])
            crop_w_arg = longitude_mask
            # print(f"crop_lot by {self.crop_lon}")
        for data_name in data_type_names:
            if self.use_cache and (data_name in self.mean_std):
                # if isinstance(self.mean_std[data_name], float):
                #     self.mean_std[data_name] = np.array([mean_std[data_name]])
                mean_std_ = self.mean_std[data_name]
                mean_std_["mean"] = float2array(mean_std_["mean"])
                mean_std_["std"] = float2array(mean_std_["std"])
                # print(np.array([mean_std[data_name]]))
                self.mean_std[data_name] = mean_std_
                mean_std[data_name] = mean_std_
                continue
            ec_data = ec_data_file[data_name]

            last_data_mean, last_data_std = \
                get_all_data_mean_std(ec_data,
                                      crop_w_arg=crop_w_arg,
                                      crop_h_arg=crop_h_arg,
                                      **self.kwargs
                                      )
            if isinstance(last_data_mean, float):
                last_data_mean = np.array([last_data_mean])
            if isinstance(last_data_std, float):
                last_data_std = np.array([last_data_std])
            self.mean_std[data_name] = dict(mean=last_data_mean, std=last_data_std)
            mean_std[data_name] = self.mean_std[data_name]
        if self.use_cache:
            with open(self.mean_std_json_pth, 'wb') as fw:
                # print(self.mean_std)
                pickle.dump(self.mean_std, fw)
            # np.save(self.mean_std_json_pth, self.mean_std, allow_pickle=True)
        return mean_std

    @property
    def sav_pth(self):
        return self.mean_std_json_pth

    @property
    def crop_args(self):
        return self.crop_h_arg, self.crop_w_arg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f"{self.kwargs}",)


def _normal(data_in, data_mean, data_std, eps=1e-4):
    """

    :param data_in:
    :param data_mean:
    :param data_std:
    :param eps:
    :return:
    """
    np.subtract(data_in, data_mean, data_in)
    stdinv = 1/(eps+data_std)
    np.multiply(data_in, stdinv, data_in)
    return data_in


@PIPELINES.register_module()
class ECNormalize:
    """Normalize images with the given mean and std value.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys" and these keys with postfix '_norm_cfg'.
    It also supports normalizing a list of images.

    Args:
        keys (Sequence[str]): The images to be normalized.
        mean (np.ndarray): Mean values of different channels.
        std (np.ndarray): Std values of different channels.
        to_rgb (bool): Whether to convert channels from BGR to RGB.
    """

    def __init__(self,
                 keys,
                 mean_std_dict=None,
                 save_original=False,
                 **get_mean_std_kwargs
                 ):
        if isinstance(mean_std_dict, str):
            with open(mean_std_dict, 'rb') as fr:
                mean_std_dict = pickle.load(fr)
        if isinstance(mean_std_dict, dict):
            self.mean_std_dict = mean_std_dict
        else:
            raise ValueError(f"'mean_std_dict' should be dic or dict.pickle file path str(PathLike)"
                             f" not {type(mean_std_dict)}")
        self.mean_std_dict = mean_std_dict
        self.save_original = save_original
        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if not self.mean_std_dict:
            return results
        results["data_mean_std"] = self.mean_std_dict
        concat_mean = []
        concat_std = []
        data_type_names = results['data_types']
        for data_name in data_type_names:
            n_mean = self.mean_std_dict[data_name]["mean"]
            n_std = self.mean_std_dict[data_name]["std"]
            n_mean = n_mean[..., None]
            n_std = n_std[..., None]
            concat_mean.append(n_mean)
            concat_std.append(n_std)
        concat_mean = np.concatenate(concat_mean, axis=-1)
        concat_std = np.concatenate(concat_std, axis=-1)
        results[f'data_mean'] = concat_mean.copy()
        results[f'data_std'] = concat_std.copy()
        for key in self.keys:
            data_frames = results[key]
            if self.save_original:
                results[f'{key}_unnormalize'] = data_frames.copy()
            _normal(data_frames, concat_mean, concat_std)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(save_original={self.save_original}, mean_std_dict={self.mean_std_dict}, '
                     f'get_mean_std={self.get_mean_std}, '
                     f')')
        return repr_str



