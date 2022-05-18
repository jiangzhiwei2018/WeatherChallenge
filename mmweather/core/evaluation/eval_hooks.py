# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import bisect
import mmcv
from mmcv.runner import Hook
from torch.utils.data import DataLoader
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import torch.distributed as dist
from mmcv.utils import TORCH_VERSION, digit_version


# from mmweather.apis import single_gpu_test


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class EvalIterHook(Hook):
    """Non-Distributed evaluation hook for iteration-based runner.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        eval_kwargs (dict): Other eval kwargs. It contains:
            save_image (bool): Whether to save image.
            save_path (str): The path to save image.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        from mmweather.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.iter)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Evaluation function.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            results (dict): Model forward results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalIterHook(EvalIterHook):
    """Distributed evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        eval_kwargs (dict): Other eval kwargs. It may contain:
            save_image (bool): Whether save image.
            save_path (str): The path to save image.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 **eval_kwargs):
        super().__init__(dataloader, interval, **eval_kwargs)
        self.gpu_collect = gpu_collect

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        from mmweather.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.iter)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)


class EvalHook(BaseEvalHook):
    def __init__(self, *args, dynamic_intervals=None,
                 tensor_board_figure_out=None,
                 tensor_board_figure_out_dir=None,
                 eval_tag="val",
                 **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)
        # assert isinstance(use_tensor_board_figure_out_dir, )
        self.tensor_board_figure_out = tensor_board_figure_out
        self.eval_tag = eval_tag
        self.tensor_board_figure_out_dir = tensor_board_figure_out_dir
        self.save_results = self.eval_kwargs.pop('save_results', False)
        self.do_evaluate = self.eval_kwargs.pop('do_evaluate', True)
        self.save_path = self.eval_kwargs.pop('save_path', None)
        self.test_val_steps = self.eval_kwargs.pop('test_val_steps', None)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def get_epoch(self, runner):
        if runner.mode == 'train':
            epoch = runner.epoch + 1
        elif runner.mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch

    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def before_run(self, runner):
        super().before_run(runner)
        if self.tensor_board_figure_out:
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

            if self.tensor_board_figure_out_dir is None:
                self.tensor_board_figure_out_dir = osp.join(runner.work_dir, f'tf_{self.eval_tag}_figure_logs')
            self.tensor_board_figure_out = SummaryWriter(self.tensor_board_figure_out_dir)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        print(f"to do {self.eval_tag} evaluate ....")
        results = self.test_fn(runner.model, self.dataloader, save_results=self.save_results,
                               save_path=self.save_path, runner=runner,
                               summary_writer=self.tensor_board_figure_out,
                               summary_writer_tag=self.eval_tag, iteration=self.get_iter(runner),
                               test_val_steps=self.test_val_steps, **self.eval_kwargs
                               )

        if not self.do_evaluate:
            return
        # self.get_triggered_stages()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)

    def after_run(self, runner):
        if self.tensor_board_figure_out is not None:
            self.tensor_board_figure_out.close()


# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.
class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)
        self.save_results = self.eval_kwargs.pop('save_results', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        # from mmdet.apis import multi_gpu_test
        results = self.test_fn(
            runner.model,
            self.dataloader, save_results=self.save_results, save_path=self.save_path, runner=runner)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
