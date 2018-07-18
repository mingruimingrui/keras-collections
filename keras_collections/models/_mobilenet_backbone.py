import keras

from ..layers import InceptionPreprocess


PREPROCESSING_LAYER = InceptionPreprocess
CUSTOM_OBJECTS = {
    'relu6': keras.applications.mobilenet.mobilenet.relu6,
    'InceptionPreprocess': InceptionPreprocess
}


def load_mobilenet_backbone(backbone='mobilenet', freeze_backbone=False):
    # TODO: Implement freeze_backbone
    assert backbone in ['mobilenet', 'mobilenetv2']

    if backbone == 'mobilenet':
        backbone_fn_name = 'MobileNet'
    else:
        backbone_fn_name = 'MobileNetV2'

    # Build original model
    mobilenet_model = getattr(keras.applications, backbone_fn_name)(
        (None, None, 3),
        alpha=1,
        include_top=False,
        weights=None
    )

    # Extract feature layers
    activation_layers = [l for l in mobilenet_model.layers if l.__class__.__name__ == 'Activation']
    C1 = activation_layers[2].output
    C2 = activation_layers[6].output
    C3 = activation_layers[10].output
    C4 = activation_layers[22].output
    C5 = activation_layers[26].output

    # Build backbone model
    mobilenet_model = keras.models.Model(
        inputs=mobilenet_model.inputs,
        outputs=[C2, C3, C4, C5],
        name='mobilenet_backbone'
    )

    # Freeze backbone layers
    batch_norm_layers = [l for l in mobilenet_model.layers if l.__class__.__name__ == 'BatchNormalization']
    for l in batch_norm_layers:
        l.trainable = False

    if freeze_backbone:
        for l in mobilenet_model.layers:
            l.trainable = False

    return mobilenet_model
