from __future__ import division
from copy import deepcopy

from ..utils.collections import AttrDict

# Define default parameters
_c = AttrDict()

################################################################################
#### Start of configurable parameters

_c.name        = 'retinanet'
_c.num_classes = None

_c.backbone        = 'resnet50'
_c.freeze_backbone = False

_c.anchor_sizes   = [32, 64, 128, 256, 512]
_c.anchor_strides = [8, 16, 32, 64, 128]
_c.anchor_ratios  = [0.5, 1., 2.]
_c.anchor_scales  = [2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)]

_c.pyramid_feature_size        = 256
_c.classification_feature_size = 256
_c.regression_feature_size     = 256

################################################################################
#### End of configurable parameters

# Set default configs to be immutable
_c.immutable(True)

def validate_config(config):
    assert isinstance(config.num_classes, int), 'num_classes must be specified'
    config.num_anchors = len(config.anchor_ratios) * len(config.anchor_scales)

def make_config(**kwargs):
    config = deepcopy(_c)
    config.immutable(False)

    # Update default config with user provided ones
    for arg, value in kwargs.items():
        config[arg] = value

    # Validate
    validate_config(config)

    config.immutable(True)

    return config
