import keras
import keras_resnet
import keras_resnet.models

from ..layers import ResNetPreprocess


PREPROCESSING_LAYER = ResNetPreprocess
CUSTOM_OBJECTS = keras_resnet.custom_objects
CUSTOM_OBJECTS['ResNetPreprocess'] = ResNetPreprocess

def load_resnet_backbone(backbone='resnet50', freeze_backbone=False):
    # TODO: Implement freeze_backbone
    assert backbone in [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'resnet200',
    ]

    # Build resnet model
    backbone_fn_name = backbone.replace('resnet', 'ResNet')
    resnet_model = getattr(keras_resnet.models, backbone_fn_name)(
        keras.Input(shape=(None, None, 3)),
        include_top=False,
        freeze_bn=True
    )

    # Freeze backbone if needed
    if freeze_backbone:
        for l in resnet_model.layers:
            l.trainable = False

    return resnet_model
