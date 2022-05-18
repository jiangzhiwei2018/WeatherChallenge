# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True, grid=None):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
        grid (Tensor): (h, w, 2)

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[-3:-1]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[-3:-1]}) are not the same.')
    _, _, h, w = x.size()
    if grid is None:
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)
    grid = grid.type_as(x)
    grid.requires_grad = False
    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[..., 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[..., 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=-1)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def get_up_feat(feat_in, up_factor, floor_ceil="floor",
                interpolation='bilinear',
                padding_mode='border',
                align_corners=False
                ):
    n, t, c, h, w = feat_in.shape
    feat = feat_in.view(-1, c, h, w)
    nt = feat.shape[0]
    # up_factor = 1.2
    # inx = torch.rand(size=(10, 5, h, w))
    up_h = int(h * up_factor)
    up_w = int(w * up_factor)
    # up_inx = torch.rand(size=(10, 5, up_h, up_w))
    arange_up_h = torch.arange(0, up_h)
    arange_up_w = torch.arange(0, up_w)
    grid_h, grid_w = torch.meshgrid(arange_up_h, arange_up_w, indexing='ij')
    grid = torch.stack((grid_w, grid_h), 2).type_as(feat)  # (h, w, 2)
    grid.requires_grad = False

    if floor_ceil == "floor":
        grid = grid.div(up_factor).floor_()
    else:
        grid = grid.div(up_factor).ceil_()
    grid = grid[None].expand(size=(nt, -1, -1, -1))
    grid_flow_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=-1)
    print(feat.shape)
    print(grid_flow.shape)
    output = F.grid_sample(
        feat,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners).view(n, t, c, up_h, up_w)
    # print(output.view(n, t, c, up_h, up_w)[0, 0, 0])
    # print(feat_in[0, 0, 0])

    return output


