# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import mmcv
import pandas as pd
from .base_weather_dataset import BaseWeatherDataset
from .registry import DATASETS
import numpy as np
from imageio import imread


def img_read(frame, factor):
    """

    :param frame:
    :param factor:
    :return:
    """
    image = np.array(imread(frame), dtype=np.uint8)[None, None]
    return image


def img_read_list(frame_list, factor=1.):
    # return frame_list
    img_list = []
    for frame in frame_list:
        img_list.append(img_read(frame, factor))
    img_list = np.concatenate(img_list, axis=0)
    return img_list


def parse_test(test_fold):
    """

    :return:
    """
    final_out = []
    file_fold_list = os.listdir(test_fold)
    for n_file_fold in file_fold_list:
        n_pth = os.path.join(test_fold, n_file_fold)
        if os.path.isdir(n_pth):
            img_file_list = sorted(os.listdir(n_pth))
            n_input = []
            for img_name in img_file_list:
                n_input.append(os.path.join(n_pth, img_name))
            final_out.append(n_input)
    return final_out


@DATASETS.register_module()
class WeatherDataset(BaseWeatherDataset):

    def __init__(self,
                 dataset_folder_name,
                 pipeline,
                 dataset_prefix=r"G:\LargeDataset\TIANCHI\weather",
                 data_type_name=("Precip", "Radar", "Wind"),
                 test_mode=False):
        super().__init__(pipeline, test_mode)
        self.dataset_folder_name = dataset_folder_name
        self.dataset_folder = os.path.join(dataset_prefix, dataset_folder_name)
        self.data_type_name = data_type_name
        self.csv_pth = os.path.join(self.dataset_folder, self.dataset_folder_name + ".csv")
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        # if self.test_mode:
        data_infos = []
        if not self.test_mode:
            csv_df = np.array(pd.read_csv(self.csv_pth, header=None))
            input_arr = csv_df[::2, :20]
            out_arr = csv_df[::2, 20:]
            data_size = len(input_arr)
            data_body = {}
            for key in self.data_type_name:
                n_data_fold = os.path.join(self.dataset_folder, key) + os.sep + f"{key}_".lower()
                data_body[key] = {"input_arr": n_data_fold + input_arr, "output_arr": n_data_fold + out_arr}
        else:
            data_body = {}
            data_size = 0
            for key in self.data_type_name:
                input_arr = parse_test(os.path.join(self.dataset_folder, key))
                data_body[key] = {"input_arr":  input_arr}
                data_size = len(input_arr)
        prog_bar = mmcv.ProgressBar(data_size)
        for i in range(data_size):
            data_info = dict(num_range=i+1)
            for key in self.data_type_name:
                n_key_data_info = data_body.get(key, None)
                assert n_key_data_info
                input_arr = n_key_data_info["input_arr"]
                output_arr = n_key_data_info.get("output_arr", None)
                data_info[key] = [img_read_list(input_arr[i]), img_read_list(output_arr[i])
                if output_arr is not None else None]
            prog_bar.update(1)
            data_infos.append(data_info)
        return data_infos
