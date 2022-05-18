# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from mmcv.cnn import build_model_from_cfg
MODELS = Registry('model', parent=MMCV_MODELS)
BACKBONES = MODELS
COMPONENTS = MODELS
LOSSES = MODELS
FLOWS = MODELS
