# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, CheckpointLoader
from mmweather.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmweather.core.distributed_wrapper import DistributedDataParallelWrapper
from mmweather.datasets import build_dataloader, build_dataset
from mmweather.models import build_model
from mmweather.utils import setup_multi_processes
from mmcv.utils import TORCH_VERSION, digit_version
from mmweather.models.backbones.weather_backbones.weather_model_backbone import BasicWeather
from mmweather.utils.misc import find_best_checkpoint

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


def main(config_pth=r"configs/final_cfg.py", save_eval_out=True):
    cfg = Config.fromfile(config_pth)
    deterministic = True
    setup_multi_processes(cfg)
    seed = 2022
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    distributed = False
    # init distributed env first, since logger depends on the dist info.
    if distributed:
        init_dist('pytorch', **cfg.dist_params)
    rank, _ = get_dist_info()
    # set random seeds
    if seed is not None:
        # if rank == 0:
        print('set random seed to', seed)
        set_random_seed(seed, deterministic=deterministic)
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    #
    # for data in data_loader:
    #     gt = data["gt"]
    #     lq = data["lq"]
    #     frameIndex = data["frameIndex"]
    #     input_dict = dict(lq=lq, frameIndex=frameIndex)
    #     print(gt.shape)
    #     print(lq.shape)
    #     print(frameIndex)
    #     model_out = model.generator(input_dict)
    #     for key, val in model_out.items():
    #         print(key)
    #         print(val.size())
    #     break
    # batch_size = 2
    # h = 8
    # w = 8
    # gt = torch.rand(size=(batch_size, 9, 1, 8*h, 8*w))
    # input_dict = dict(lq=torch.rand(size=(batch_size, 2, 1, h, w)), frameIndex=torch.rand(size=(batch_size, 7)))
    # model_out = model.generator(input_dict)
    # for key, val in model_out.items():
    #     print(key)
    #     print(val.size())
    # return
    work_dir = cfg.get("work_dir", "./")
    checkpoint = find_best_checkpoint(work_dir)
    assert checkpoint is not None
    test_save_dir = os.path.join(work_dir, "test_results")
    os.makedirs(test_save_dir, exist_ok=True)
    if not distributed:
        out_checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model,
            data_loader,
            save_path=test_save_dir,
            save_results=True,
            write_img=True,
            summary_writer=None,
            summary_writer_tag="test_final", iteration=0,
            test_val_steps=None
        )
        return
    else:
        raise ValueError("only support non distributed")
    if rank == 0 and 'eval_result' in outputs[0]:
        # print metrics
        stats = dataset.evaluate(outputs)
        for stat in stats:
            print('Eval-{}: {}'.format(stat, stats[stat]))
        # save result pickle
        if save_eval_out:
            eval_result_save_dir = os.path.join(work_dir, "test_results", "eval_result")
            os.makedirs(eval_result_save_dir, exist_ok=True)
            eval_result_save_pth = os.path.join(eval_result_save_dir, "eval_dict.pkl")
            print('writing results to {}'.format(eval_result_save_pth))
            mmcv.dump(outputs, eval_result_save_pth)


if __name__ == '__main__':
    main()
    # o = dict(a=1, b=2)
    # mmcv.dump(o, "./aaaa.json")
    # s = mmcv.load("./aaaa.json")
    # print(s)
