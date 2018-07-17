""" Backbone loader """


def load_backbone(backbone='resnet50', freeze_backbone=False):
    if 'resnet' in backbone:
        from ._resnet_backbone import load_resnet_backbone as _load_backbone
    else:
        raise Exception('Backbone name {} has not been implemented'.format(backbone))

    return _load_backbone(backbone, freeze_backbone=freeze_backbone)


def load_backbone_preprocessing(backbone='resnet50'):
    if 'resnet' in backbone:
        from ._resnet_backbone import PREPROCESSING_LAYER as preprocessing_layer
    return preprocessing_layer()


def load_backbone_custom_objects(backbone='resnet50'):
    if 'resnet' in backbone:
        from ._resnet_backbone import CUSTOM_OBJECTS as custom_objects
    else:
        raise Exception('Backbone name {} has not been implemented'.format(backbone))

    return custom_objects
