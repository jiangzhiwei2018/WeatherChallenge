# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from mmcv.utils import TORCH_VERSION, digit_version
import mmcv
import torch
from mmcv.runner import HOOKS, Hook, TensorboardLoggerHook
from mmcv.runner.dist_utils import master_only
from torchvision.utils import save_image


@HOOKS.register_module()
class MyCheckPointHook(Hook):
    """Visualization hook.

    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.

    Args:
        output_dir (str): The file path to store visualizations.
        res_name_list (str): The list contains the name of results in outputs
            dict. The results in outputs dict must be a torch.Tensor with shape
            (n, c, h, w).
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
        rerange (bool): Whether to rerange the output value from [-1, 1] to
            [0, 1]. We highly recommend users should preprocess the
            visualization results on their own. Here, we just provide a simple
            interface. Default: True.
        bgr2rgb (bool): Whether to reformat the channel dimension from BGR to
            RGB. The final image we will save is following RGB style.
            Default: True.
        nrow (int): The number of samples in a row. Default: 1.
        padding (int): The number of padding pixels between each samples.
            Default: 4.
    """

    def __init__(self,
                 output_dir,
                 res_name_list,
                 interval=-1,
                 filename_tmpl='iter_{}.png',
                 rerange=True,
                 bgr2rgb=True,
                 nrow=1,
                 padding=4):
        assert mmcv.is_list_of(res_name_list, str)
        self.output_dir = output_dir
        self.res_name_list = res_name_list
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.bgr2rgb = bgr2rgb
        self.rerange = rerange
        self.nrow = nrow
        self.padding = padding

        mmcv.mkdir_or_exist(self.output_dir)

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        results = runner.outputs['results']

        filename = self.filename_tmpl.format(runner.iter + 1)

        img_list = [x for k, x in results.items() if k in self.res_name_list]
        img_cat = torch.cat(img_list, dim=3).detach()
        if self.rerange:
            img_cat = ((img_cat + 1) / 2)
        if self.bgr2rgb:
            img_cat = img_cat[:, [2, 1, 0], ...]
        img_cat = img_cat.clamp_(0, 1)
        save_image(
            img_cat,
            osp.join(self.output_dir, filename),
            nrow=self.nrow,
            padding=self.padding)


@HOOKS.register_module()
class MyTensorBoardHook(TensorboardLoggerHook):
    def __init__(self, *args, **kwargs):
        super(TensorboardLoggerHook).__init__(*args, **kwargs)
        self.writer = None

    @master_only
    def before_run(self, runner):
        super(TensorboardLoggerHook, self).before_run(runner)
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

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()






