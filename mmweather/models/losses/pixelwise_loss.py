# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmweather.models.common.flow_warp import flow_warp
from .perceptual_loss import PerceptualLoss
from ..registry import LOSSES
from .utils import masked_loss

_reduction_modes = ['none', 'mean', 'sum']


@masked_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


@masked_loss
def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target) ** 2 + eps)


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class MaskedTVLoss(L1Loss):
    """Masked TV loss.

        Args:
            loss_weight (float, optional): Loss weight. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def forward(self, pred, mask=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor, optional): Tensor with shape of (n, 1, h, w).
                Defaults to None.

        Returns:
            [type]: [description]
        """
        y_diff = super().forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :], weight=mask[:, :, :-1, :])
        x_diff = super().forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:], weight=mask[:, :, :, :-1])

        loss = x_diff + y_diff

        return loss


@LOSSES.register_module()
class MyPixelWiseLossVFI(nn.Module):
    def __init__(self, layer_weights={'4': 1., '9': 1., '18': 1.},
                 H=None, W=None, pretrained='./vgg19-dcbb9e9d.pth', vgg_type="vgg19"):
        super(MyPixelWiseLossVFI, self).__init__()
        self.L1_lossFn = L1Loss()
        self.MSE_LossFn = MSELoss()
        # self.mseLoss = MSELoss(loss_weight=0.05)
        if H and W:
            # create a grid
            H = int(H)
            W = int(W)
            grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            self.grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)
        else:
            self.grid = None
        self.pre_loss = PerceptualLoss(layer_weights=layer_weights, pretrained=pretrained, vgg_type=vgg_type,
                                       use_input_norm=False, norm_img=False)

    def forward(self, result_output, gt):
        """

        :param result_output:
        :return:
        """
        # flow_warp()
        Ft_p = result_output['Ft_p']
        I0 = result_output['I0']
        I1 = result_output['I1']
        g_I0_F_t_0 = result_output['g_I0_F_t_0']
        g_I1_F_t_1 = result_output['g_I1_F_t_1']
        F_1_0 = result_output['F_1_0']
        F_0_1 = result_output['F_0_1']

        n, frame_size, c, h, w = Ft_p.shape
        vfi_gt = gt[:, 1:-1]
        if self.grid is None:
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
            self.grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)

        _, _, c, uph, upw = vfi_gt.shape
        Ft_p = Ft_p.view(-1, c, h, w)
        IFrame = vfi_gt.view(-1, c, uph, upw)
        # IFrame = F.interpolate(IFrame, size=(h, w), mode="bicubic", align_corners=True)  # (-1, c, h, w)
        recnLoss = self.L1_lossFn(Ft_p, IFrame)
        percep_loss, style_loss = self.pre_loss(Ft_p.expand(-1, 3, h, w), IFrame.expand(-1, 3, h, w))

        warpLoss = self.L1_lossFn(g_I0_F_t_0, IFrame) + self.L1_lossFn(g_I1_F_t_1, IFrame) + \
                   self.L1_lossFn(flow_warp(I0, F_1_0, grid=self.grid), I1) \
                   + self.L1_lossFn(flow_warp(I1, F_0_1, grid=self.grid), I0)

        # loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[..., :-1] - F_1_0[..., 1:])) + torch.mean(
        #     torch.abs(F_1_0[..., :-1, :] - F_1_0[..., 1:, :]))
        # loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[..., :-1] - F_0_1[..., 1:])) + torch.mean(
        #     torch.abs(F_0_1[..., :-1, :] - F_0_1[..., 1:, :]))
        # loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
        loss = 0.8 * recnLoss + 0.2 * warpLoss + percep_loss
        return loss


@LOSSES.register_module()
class MyPixelWiseLoss(nn.Module):
    def __init__(self, H=None, W=None):
        super(MyPixelWiseLoss, self).__init__()
        self.L1_lossFn = L1Loss()
        self.MSE_LossFn = MSELoss()
        # self.mseLoss = MSELoss(loss_weight=0.05)
        if H and W:
            # create a grid
            H = int(H)
            W = int(W)
            grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            self.grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)
        else:
            self.grid = None

    def forward(self, result_output, gt):
        """

        :param result_output:
        :return:
        """
        # flow_warp()
        Ft_p = result_output['Ft_p']
        # lq = result_output['lq']
        I0 = result_output['I0']
        I1 = result_output['I1']
        g_I0_F_t_0 = result_output['g_I0_F_t_0']
        g_I1_F_t_1 = result_output['g_I1_F_t_1']
        F_1_0 = result_output['F_1_0']
        F_0_1 = result_output['F_0_1']

        n, frame_size, c, h, w = Ft_p.shape
        vfi_gt = gt[:, 1:-1]
        if self.grid is None:
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
            self.grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)

        _, _, c, uph, upw = vfi_gt.shape
        Ft_p = Ft_p.view(-1, c, h, w)
        IFrame = vfi_gt.view(-1, c, uph, upw)

        IFrame = F.interpolate(IFrame, size=(h, w), mode="bicubic", align_corners=True)  # (-1, c, h, w)

        recnLoss = self.L1_lossFn(Ft_p, IFrame)
        prcpLoss = self.MSE_LossFn(Ft_p, IFrame)

        warpLoss = self.L1_lossFn(g_I0_F_t_0, IFrame) + self.L1_lossFn(g_I1_F_t_1, IFrame) + \
                   self.L1_lossFn(flow_warp(I0, F_1_0, grid=self.grid), I1) \
                   + self.L1_lossFn(flow_warp(I1, F_0_1, grid=self.grid), I0)

        loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[..., :-1] - F_1_0[..., 1:])) + torch.mean(
            torch.abs(F_1_0[..., :-1, :] - F_1_0[..., 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[..., :-1] - F_0_1[..., 1:])) + torch.mean(
            torch.abs(F_0_1[..., :-1, :] - F_0_1[..., 1:, :]))
        loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth + \
               204 * self.L1_lossFn(gt, result_output['output'])
        return loss

# @LOSSES.register_module()
# class L1_Charbonnier_loss(torch.nn.Module):
#     """L1 Charbonnierloss."""
#     def __init__(self):
#         super(L1_Charbonnier_loss, self).__init__()
#         self.eps = 1e-6
#
#     def forward(self, X, Y):
#         diff = torch.add(X, -Y)
#         error = torch.sqrt(diff * diff + self.eps)
#         loss = torch.mean(error)
#         return loss
