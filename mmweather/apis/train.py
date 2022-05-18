# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel
from mmcv.runner import HOOKS, IterBasedRunner, get_dist_info, EpochBasedRunner, build_runner, Fp16OptimizerHook, \
    OptimizerHook, DistSamplerSeedHook, CheckpointHook, IterTimerHook, TextLoggerHook, \
    CosineAnnealingLrUpdaterHook, TensorboardLoggerHook


from mmcv.utils import build_from_cfg
from mmweather.apis import single_gpu_test, multi_gpu_test
from mmweather.core import DistEvalIterHook, EvalIterHook, build_optimizers
from mmweather.core.distributed_wrapper import DistributedDataParallelWrapper
from mmweather.core.evaluation.eval_hooks import DistEvalHook, EvalHook
from mmweather.datasets.builder import build_dataloader, build_dataset
from mmweather.utils import get_root_logger
from mmweather.utils.misc import find_latest_checkpoint


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.
    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                test=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=distributed,
            num_gpus=len(cfg.gpu_ids),
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False)
        ),
        **({} if torch.__version__ != 'parrots' else dict(
            prefetch_num=2,
            pin_memory=False,
        )),
        **dict((k, cfg.data[k]) for k in [
            'samples_per_gpu',
            'workers_per_gpu',
            'shuffle',
            'seed',
            'drop_last',
            'prefetch_num',
            'pin_memory',
        ] if k in cfg.data)
    }
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = DistributedDataParallelWrapper(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer)
    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None)  # user-defined hooks
    )

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    # evaluation hook
    if validate and cfg.get('evaluation', None) is not None:
        dataset = build_dataset(cfg.data.val)
        if ('val_samples_per_gpu' in cfg.data
                or 'val_workers_per_gpu' in cfg.data):
            warnings.warn('"val_samples_per_gpu/val_workers_per_gpu" have '
                          'been deprecated. Please use '
                          '"val_dataloader=dict(samples_per_gpu=1)" instead. '
                          'Details see '
                          'https://github.com/open-mmlab/mmweathering/pull/201')

        val_loader_cfg = {
            **loader_cfg,
            **dict(shuffle=False, drop_last=False, dist=distributed),
            **dict((newk, cfg.data[oldk]) for oldk, newk in [
                ('val_samples_per_gpu', 'samples_per_gpu'),
                ('val_workers_per_gpu', 'workers_per_gpu'),
            ] if oldk in cfg.data),
            **cfg.data.get('val_dataloader', {})
        }
        val_dataloader = build_dataloader(dataset, **val_loader_cfg)
        eval_cfg = cfg.get('evaluation', {})

        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        test_fn = multi_gpu_test if distributed else single_gpu_test
        eval_cfg["test_fn"] = test_fn
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if test and cfg.get('test_evaluation', None) is not None and cfg.data.get("test", None) is not None:
        print("to do test...........")
        dataset = build_dataset(cfg.data.test)
        if ('test_samples_per_gpu' in cfg.data
                or 'test_workers_per_gpu' in cfg.data):
            warnings.warn('"test_samples_per_gpu/test__workers_per_gpu" have '
                          'been deprecated. Please use '
                          '"test_dataloader=dict(samples_per_gpu=1)" instead. '
                          'Details see '
                          'https://github.com/open-mmlab/mmweathering/pull/201')
        test_loader_cfg = {
            **loader_cfg,
            **dict(shuffle=False, drop_last=False, dist=distributed),
            **dict((newk, cfg.data[oldk]) for oldk, newk in [
                ('test_samples_per_gpu', 'samples_per_gpu'),
                ('test_workers_per_gpu', 'workers_per_gpu'),
            ] if oldk in cfg.data),
            **cfg.data.get('test_dataloader', {})
        }
        test_dataloader = build_dataloader(dataset, **test_loader_cfg)
        eval_cfg = cfg.get('test_evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        test_fn = multi_gpu_test if distributed else single_gpu_test
        eval_cfg["test_fn"] = test_fn
        runner.register_hook(
            eval_hook(test_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
