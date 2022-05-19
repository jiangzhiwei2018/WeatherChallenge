import segmentation_models_pytorch as spm
from mmweather.models.registry import BACKBONES
from segmentation_models_pytorch import *

BACKBONES.register_module('SegPSPNet', module=PSPNet)
BACKBONES.register_module('SegUnet', module=Unet)
BACKBONES.register_module('SegUnetPlusPlus', module=UnetPlusPlus)
BACKBONES.register_module('SegDeepLabV3', module=DeepLabV3)
BACKBONES.register_module('SegDeepLabV3Plus', module=DeepLabV3Plus)
BACKBONES.register_module('SegLinknet', module=Linknet)
BACKBONES.register_module('SegMAnet', module=MAnet)
BACKBONES.register_module('SegPAN', module=PAN)
BACKBONES.register_module('SegFPN', module=FPN)








