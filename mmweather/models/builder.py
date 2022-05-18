# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv import build_from_cfg
from mmcv.cnn import NORM_LAYERS
from mmcv.cnn.bricks.norm import infer_abbr

from .registry import BACKBONES, COMPONENTS, LOSSES, MODELS, FLOWS


def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone.

    Args:
        cfg (dict): Configuration for building backbone.
    """
    return build(cfg, BACKBONES)


def build_flow(cfg):
    """Build backbone.

    Args:
        cfg (dict): Configuration for building flow compute.
    """
    return build(cfg, FLOWS)


def build_component(cfg):
    """Build component.

    Args:
        cfg (dict): Configuration for building component.
    """
    return build(cfg, COMPONENTS)


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (dict): Configuration for building loss.
    """
    return build(cfg, LOSSES)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_norm(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        nn.Module: norm layer.
    """
    if cfg is None:
        return nn.Identity()
    if not isinstance(cfg, dict):

        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer

