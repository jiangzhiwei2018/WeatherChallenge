from bisect import bisect_left
from datetime import datetime, timedelta
import gc
import os
from string import ascii_lowercase
from typing import Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec
import netCDF4
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
# from utils_src import utils, save_emf
import seaborn as sns
from matplotlib import cm as CM
# from py_src import eval_api
# matplotlib.use("Agg")
import json
import torch.nn.functional as F


def plot_img(img, value_range=(np.log10(0.1), np.log10(100)), extent=None):
    plt.imshow(img, interpolation='nearest',
               norm=colors.Normalize(*value_range),
               extent=extent,
               cmap=CM.RdBu_r)  # , vmin=value_range[0], vmax=value_range[1])
    # sns.heatmap(img, cmap="RdBu_r", center=0, vmin=value_range[0], vmax=value_range[1])
    # plt.imshow(img, interpolation='nearest', extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)


def plot_single_batch(lq, start_frame_idx, frameIdx,
                      gt=None, compare_out=None, application="default",
                      out_file_format="svg", save_pth=None, value_range=None, channel_idx=0,
                      start_date='2022-01-01',
                      date_format='%Y-%m-%d', units="", cmap=CM.RdBu_r, show_fig=False):
    """

    :return:
    """
    # assert lq.dim() == 4
    org_samples = 1 if gt is None else 2
    num_frames = len(frameIdx) + 2
    if compare_out is None:
        compare_out = dict()
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, date_format)
    assert isinstance(start_date, datetime)
    if value_range is None:
        value_range = (lq.min(), lq.max())
    now_date = start_date + timedelta(hours=int(start_frame_idx))
    num_rows = 1
    num_cols = num_frames
    num_rows_s = org_samples + len(compare_out)

    figsize = (num_cols * 1.5, num_rows * num_rows_s * 1.5 + 0.8)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_rows + 1, 1, hspace=0.05,
                           height_ratios=[1] * num_rows + [0.035],
                           # width_ratios=[1.]
                           )
    gs_s = gridspec.GridSpecFromSubplotSpec(num_rows_s, num_cols,
                                            subplot_spec=gs[0, 0], wspace=0.05, hspace=0.05)
    # print(start_date)
    # print(int(start_frame_idx))
    # print(timedelta(hours=int(start_frame_idx)))

    now_date_title_str = now_date.strftime('%Y-%m-%d_%H%M%S')
    # print(gs_s[1, 1])
    for t in range(num_frames):
        is_head_end = t == 0 or t == num_frames - 1
        if is_head_end:
            n_timedelta = t
        else:
            n_timedelta = float(frameIdx[t - 1])
        n_hour_date = now_date + timedelta(hours=n_timedelta)
        if t == 0:
            plt.subplot(gs_s[0, 0])
            # plt.axis('square')
            plot_img(lq[0, channel_idx, :, :], value_range=value_range)
            plt.ylabel("LQ", fontsize=16)
        if t == num_frames - 1:
            plt.subplot(gs_s[0, -1])
            # plt.axis('square')
            plot_img(lq[-1, channel_idx, :, :], value_range=value_range)
        plt.subplot(gs_s[0, t])
        # plt.axis('square')
        plt.gca().tick_params(left=False, bottom=False,
                              labelleft=False, labelbottom=False)
        plt.title("%02d:%02d:%02d" % (n_hour_date.hour, n_hour_date.minute, n_hour_date.second),
                  fontsize=16)

        next_i = 1
        if gt is not None:
            plt.subplot(gs_s[next_i, t])
            plot_img(gt[t, channel_idx, :, :], value_range=value_range)
            if t == 0:
                plt.ylabel("GT", fontsize=16)
            next_i = next_i + 1
        for key, val in compare_out.items():
            plt.subplot(gs_s[next_i, t])
            plot_img(val[t, channel_idx, :, :], value_range=value_range)
            if t == 0:
                plt.ylabel(key, fontsize=16)
            next_i = next_i + 1

    units = units
    cb_tick_loc = [value_range[0], (value_range[0] + value_range[1]) / 2., value_range[1]]
    cb_tick_labels = np.array(cb_tick_loc).round().astype(int)
    cax = plt.subplot(gs[-1, 0]).axes
    cb = colorbar.ColorbarBase(cax, norm=colors.Normalize(*value_range),
                               orientation='horizontal', cmap=cmap)
    cb.set_ticks(cb_tick_loc)
    cb.set_ticklabels(cb_tick_labels)
    cax.tick_params()
    cb.set_label(units)
    # plt.title(f"{now_date_title_str}", fontsize=16)
    fig.suptitle(f"{now_date_title_str}", fontsize=16, va="top")
    if save_pth is not None:
        save_pth = os.path.join(save_pth, application)
        os.makedirs(save_pth, exist_ok=True)
        save_path = os.path.join(save_pth, f"{application}_{now_date_title_str}.{out_file_format}")
        plt.savefig(save_path)
    if show_fig:
        plt.show()
    # plt.close()
    now_date_tag = now_date.strftime('%Y-%m-%d/%H:%M:%S')
    return fig, now_date_tag


def interpolate_lq(lq_input, out_size_hw=(64, 64), out_size_frames=9, interpolate_mode="bicubic"):
    """

    :param lq_input:(n, t, c, h, w)
    :param out_size:
    :return:
    """
    n, t, c, h, w = lq_input.size()
    # lq_input = lq_input.view(-1, c, h, w)
    lq_hd = F.interpolate(lq_input.view(-1, c, h, w), size=out_size_hw, mode=interpolate_mode,
                          align_corners=True).view(n, t, c,
                                                   out_size_hw[0],
                                                   out_size_hw[1]
                                                   )
    final_res = []
    for nb in range(n):
        all_cn_data = []
        for n_channel in range(c):
            n_c_in = lq_hd[nb, :, n_channel].permute(1, 2, 0)
            # torch.Tensor()
            n_c_in = F.interpolate(n_c_in, size=(out_size_frames,), mode="linear",
                                   align_corners=True).permute(2, 0, 1)[None, :, None]
            # print(n_c_in.shape)
            all_cn_data.append(n_c_in)
        all_cn_data = torch.cat(all_cn_data, dim=2)
        final_res.append(all_cn_data)
    final_res = torch.cat(final_res, dim=0)
    return final_res


def plot_eval_result(lq, gt, start_frame_idx, frameIdx, compare_out=None,
                     out_file_format="svg", save_pth=None,
                     out_size_hw=(64, 64), out_size_frames=9, interpolate_modes=["bicubic", "bilinear"],
                     datetime_str='2022-01-01',
                     date_format='%Y-%m-%d', show_fig=False,
                     start_year_month_date=None,
                     **kwargs
                     ):
    """

    :return:
    """
    # assert start_year_month_date is not None
    assert out_size_frames == frameIdx.shape[1] + 2
    if compare_out is None:
        compare_out = dict()
    batch_size = len(lq)
    assert batch_size == len(start_frame_idx)
    date_format2 = '%Y-%m-%d'
    start_date = datetime.strptime(datetime_str, date_format)
    if gt is not None:
        gt = gt.cpu()
        out_size_hw = (gt.shape[-2], gt.shape[-1])
        assert out_size_frames == gt.shape[1]

    for interpolate_mode in interpolate_modes:
        n_lq = interpolate_lq(lq, interpolate_mode=interpolate_mode, out_size_hw=out_size_hw,
                              out_size_frames=out_size_frames)
        compare_out.update({interpolate_mode: n_lq})
        # mode_lq[interpolate_mode] = n_lq
    compare_out_list = []
    for i in range(batch_size):
        compare_out_ = dict()
        for name, val in compare_out.items():
            compare_out_[name] = val[i]
        compare_out_list.append(compare_out_)
    fig_list = []
    all_tags = []
    for i in range(batch_size):
        fig, now_date_tag = plot_single_batch(lq=lq[i],
                                              start_frame_idx=start_frame_idx[i],
                                              frameIdx=frameIdx[i],
                                              show_fig=show_fig,
                                              gt=gt[i],
                                              compare_out=compare_out_list[i], out_file_format=out_file_format,
                                              save_pth=save_pth,
                                              start_date=start_year_month_date[i]['start_year_month_date'],
                                              **kwargs
                                              )
        all_tags.append(now_date_tag)
        fig_list.append(fig)
    return fig_list, all_tags


if __name__ == '__main__':
    from mmcv.utils import TORCH_VERSION, digit_version

    if (TORCH_VERSION == 'parrots'
            or digit_version(TORCH_VERSION) < digit_version('1.1')):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorboardX to use '
                              'TensorboardLoggerHook.')
    else:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install future tensorboard" to install '
                'the dependencies to use torch.utils.tensorboard '
                '(applicable to PyTorch 1.1 or higher)')

    lq = torch.rand(size=(1, 2, 1, 8, 8))
    gt = torch.rand(size=(1, 9, 1, 64, 64))
    compare_out = {"Model": torch.rand(size=(1, 9, 1, 64, 64))}
    frameIdx = torch.arange(1, 8)[None]
    # frameIdx = torch.sort(frameIdx, dim=-1)[]
    fi_list = plot_eval_result(lq, gt, start_frame_idx=torch.randint(low=0, high=1000, size=(1, 1)),
                               frameIdx=frameIdx, show_fig=True, compare_out=compare_out)

    # lq = F.interpolate(lq, size=(64, 64), mode="bicubic")
    # hq = F.interpolate(lq, size=(64, 64), mode="bicubic")
    # hq = F.interpolate(lq, size=(64, 64), mode="bicubic")

    # start_date = '2022-01-01'
    # date_format = '%Y-%m-%d'
    # start_date = datetime.strptime(start_date, date_format)
    # now_date = start_date + timedelta(hours=52.56)
    # print(now_date)
    # print(now_date.hour)
    # print(now_date.second)
    # print(now_date.minute)
