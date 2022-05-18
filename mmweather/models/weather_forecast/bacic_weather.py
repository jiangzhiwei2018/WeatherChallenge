# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
import pickle
import numpy as np
import torch
import os
import mmcv
import torch.nn as nn
from mmcv.runner import auto_fp16
# from mmcv.utils import build_from_cfg
# from mmedit.core import psnr, ssim,
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from ...core.evaluation.my_eval_metrics import PSNR, SSIM
from mmcv.utils import TORCH_VERSION, digit_version
from imageio import imwrite

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


@MODELS.register_module()
class BasicWeatherModel(BaseModel):
    """Basic model for image restoration.

    It must contain a generator that takes an image as inputs and outputs a
    restored image. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 val_range=(0, 255),
                 eps=1e-4,
                 image_write_keys_order=("Precip", "Radar", "Wind"),
                 image_write_prefix="./submit"
                 ):
        super().__init__()
        self.eps = eps

        # self.whether_denormalize = whether_denormalize
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.val_range = val_range
        self.allowed_metrics = {'MSE': nn.MSELoss()}
        # support fp16
        self.fp16_enabled = False
        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)
        # loss
        self.pixel_loss = build_loss(pixel_loss)
        self.image_write_keys_order = image_write_keys_order
        self.image_write_prefix = image_write_prefix
        # self.data_type_factor = {"Precip": 10, "Radar": 70, "Wind": 35}

    def denormalize(self, data):
        """

        :param data:
        :return:
        """
        if data is None:
            return data
        return torch.clip(data, 0, 1) * 255

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    def combine_image(self, data_dict):
        if data_dict is None:
            return data_dict
        data_ = []
        for key in self.image_write_keys_order:
            data_.append(data_dict[key])
        data_ = torch.cat(data_, dim=2)
        return data_

    def image_write(self, image, meta, save_pth=None):
        if save_pth is None:
            save_pth = "./submit"
        image = image.detach().cpu().numpy().astype(np.uint8)
        for i, key in enumerate(self.image_write_keys_order):
            for num in range(image.shape[0]):
                image_prefix = os.path.join(save_pth, key, f"{meta[num]['num_range']}".zfill(3))
                os.makedirs(image_prefix, exist_ok=True)
                for t in range(20):
                    img = image[num, t, i]
                    img_pth = osp.join(image_prefix, f"{key.lower()}_{str(t + 1).zfill(3)}.png")
                    imwrite(img_pth, img)

    def image_write_TensorBoard(self, image, meta, summary_writer: SummaryWriter, summary_writer_tag, iteration=None):
        for i, key in enumerate(self.image_write_keys_order):
            for num in range(image.shape[0]):
                for t in range(20):
                    tag = summary_writer_tag + "/" + f"{key}/" + str(meta[num]['num_range']).zfill(3) + "/" + \
                          f"{key.lower()}_{str(t + 1).zfill(3)}.png"
                    img = image[num, t, i]
                    summary_writer.add_image(img_tensor=img, tag=tag, global_step=iteration, dataformats="HW")

    @auto_fp16(apply_to=('input_img',))
    def forward(self, input_img, gt=None, test_mode=False, return_loss=True, **kwargs):
        """Forward function.

        Args:
            input_img (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if test_mode or not return_loss:
            return self.forward_test(input_img, gt, **kwargs)

        return self.forward_train(input_img, gt)

    def forward_train(self, input_img, gt):
        """Training forward function.

        Args:
            input_img (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        forward_input_dict = input_img
        losses = dict()
        output = self.generator(forward_input_dict)
        if gt is None:
            return output['output']
        loss_pix = self.pixel_loss(output['output'], gt)
        losses['loss_pix'] = loss_pix
        # for n_obj in forward_input_dict:
        #     if isinstance(forward_input_dict[n_obj], torch.TensorType):
        #         forward_input_dict[n_obj].cpu()
        output = output["output"]
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(
                lq=forward_input_dict.cpu(),
                gt=gt.cpu(),
                output=output.cpu()))
        return outputs

    def evaluate(self, output, gt, denormalize=True):
        """Evaluation function.

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        assert output.shape == gt.shape
        if denormalize:
            output = self.denormalize(output)
            gt = self.denormalize(gt)
        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = float(self.allowed_metrics[metric](output, gt))
        return eval_result

    # @torch.no_grad()
    def forward_test(self,
                     input_img,
                     gt: dict = None,
                     meta=None,
                     write_tensor_board=True,
                     write_img=False,
                     save_path=None,
                     save_results=False,
                     iteration=None,
                     denormalize=True,
                     start_frame=None,
                     summary_writer=None,
                     summary_writer_tag="default_forward_test",
                     start_year_month_date=None,
                     **kwargs):
        """Testing forward function.


        Returns:
            dict: Output results.
        """
        # print(meta)
        forward_input_dict = input_img
        output_dict = self.generator(forward_input_dict)
        output = output_dict["output"]
        if denormalize:
            output = self.denormalize(output)
            gt = self.denormalize(gt)

        if self.test_cfg is not None and self.test_cfg.get("metrics", None) and gt is not None:
            results = dict(eval_result=self.evaluate(output, gt, denormalize=False))
        else:
            results = dict(eval_result=None)
        if write_img and meta[0].get('num_range', None):
            self.image_write(output, meta, save_pth=save_path)
        if isinstance(summary_writer, SummaryWriter) and meta is not None:
            self.image_write_TensorBoard(output, meta, summary_writer_tag=summary_writer_tag,
                                         summary_writer=summary_writer)
        results.update(dict(batch_size=len(output)))
        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output


