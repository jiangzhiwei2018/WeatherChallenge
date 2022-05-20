import argparse
import copy
import os
import os.path as osp
import time
import warnings
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import init_dist, StepLrUpdaterHook, CosineRestartLrUpdaterHook, TextLoggerHook, CheckpointHook, \
    OptimizerHook, TensorboardLoggerHook
from mmweather import __version__
from mmweather.apis import init_random_seed, set_random_seed, train_model
from mmweather.datasets import build_dataset, build_dataloader
from mmweather.models import build_model
from mmweather.models.backbones.weather_backbones.weather_model_backbone import BasicWeather
from mmweather.utils import collect_env, get_root_logger, setup_multi_processes


# TextLoggerHook
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
def parse_args():
    parser = argparse.ArgumentParser(description='AI weather')
    parser.add_argument('--dataset_prefix', help='dataset root', default=r"G:/LargeDataset/TIANCHI/weather")
    parser.add_argument('--seed', help='seed', default=2022)
    parser.add_argument('--samples_per_gpu', help='samples_per_gpu', default=2, type=int)
    parser.add_argument('--workers_per_gpu', help='workers_per_gpu', default=4, type=int)
    args = parser.parse_args()
    return args


def main(config_pth=r"configs/final_cfg.py"):
    """

    :return:
    """
    args = parse_args()
    cfg = Config.fromfile(config_pth)
    dataset_prefix = args.dataset_prefix
    cfgdata = cfg.data
    cfgdata["train_dataloader"]["samples_per_gpu"] = args.samples_per_gpu
    cfgdata["workers_per_gpu"] = args.workers_per_gpu
    for mod in ('train', 'val', 'test'):
        if cfgdata.get(mod, None) is not None:
            cfgdata[mod]["dataset_prefix"] = dataset_prefix
    cfg.data = cfgdata
    # datasets = build_dataset(cfg.data.train)
    # return
    # data_ld = build_dataloader(datasets,
    #                            samples_per_gpu=4,
    #                            workers_per_gpu=1, shuffle=False)
    # work_dir = os.path.join("./work_dirs")
    # return
    setup_multi_processes(cfg)
    deterministic = True
    seed = args.seed
    diff_seed = False
    distributed = False
    torch.backends.cudnn.benchmark = deterministic
    cfg.gpus = len(cfg.gpu_ids)
    # distributed = True
    # cfg.dist_params = dict(backend='gloo')
    # init_dist(launcher="pytorch", **cfg.dist_params)
    # create work_dir
    # os.environ['RANK'] = '1'
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    log_file = None
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    env_info_dict = collect_env.collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'

    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('mmweather Version: {}'.format(__version__))
    logger.info('Config:\n{}'.format(cfg.text))
    # set random seeds
    seed = init_random_seed(seed)
    seed = seed + dist.get_rank() if diff_seed else seed
    logger.info('Set random seed to {}, deterministic: {}'.format(seed, deterministic))
    set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    model = build_model(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    datasets = [build_dataset(cfg.data.train)]

    # print(len(datasets[0]))
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmweather_version=__version__
            # config=cfg.text,
        )
    # meta information
    meta = dict()
    if cfg.get('exp_name', None) is None:
        cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
    meta['exp_name'] = cfg.exp_name
    meta['mmweather Version'] = __version__
    meta['seed'] = seed
    meta['env_info'] = env_info
    # print(model)
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        test=True,
        timestamp=timestamp,
        meta=meta)
    # CosineRestartLrUpdaterHook


import segmentation_models_pytorch as smp


if __name__ == '__main__':
    # m = smp.PAN(
    #     in_channels=60,
    #     classes=60,
    #     encoder_weights=None,
    #     upsampling=0
    # ).cuda()
    # inx = torch.rand(size=(8, 60, 480, 560)).cuda()
    # o = m(inx)
    # print(o.shape)
    main()
    # im = np.random.random(size=(1, ))
    # forward_input_dict = {
    #     "Precip": torch.rand(size=(4, 20, 1, 480, 560)).cuda(),
    #     "Radar": torch.rand(size=(4, 20, 1, 480, 560)).cuda(),
    #     "Wind": torch.rand(size=(4, 20, 1, 480, 560)).cuda()
    # }
    # m = BasicWeather(generator_backbone_cfg=dict(type="UnetGenerator"),
    #                  upsample_type="interpolate", conv_cfg=dict(type='DCNv2')).cuda()
    # print(m)
    # out = m(forward_input_dict)
    # print(out.size())




