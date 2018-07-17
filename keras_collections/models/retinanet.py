""" Keras Retinanet from https://github.com/fizyr/keras-retinanet
Some slight refactoring are done to improve reusability of codebase
"""

import keras

from .. import initializers
from .. import layers
from .. import losses

from ._retinanet_config import make_config
from ._retinanet import (
    default_classification_model,
    default_regression_model,
    create_pyramid_features,
    apply_model_to_features,
    compute_anchors
)
from ._load_backbone import (
    load_backbone,
    load_backbone_preprocessing,
    load_backbone_custom_objects
)


def compile_retinanet(
    training_model,
    huber_sigma=3.0,
    focal_alpha=0.25,
    focal_gamma=2.0,
    optimizer=None
):
    if optimizer is None:
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)

    training_model.compile(
        loss={
            'regression'    : losses.make_detection_huber_loss(sigma=huber_sigma),
            'classification': losses.make_detection_focal_loss(alpha=focal_alpha, gamma=focal_gamma)
        },
        optimizer=optimizer
    )


def RetinaNetLoad(filepath, backbone='resnet50'):
    """ Loads a retinanet model from a file

    Args
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone : Backbone with which the model was trained.
    """
    custom_objects = {
        'PriorProbability'     : initializers.PriorProbability,
        'ResizeTo'             : layers.ResizeTo,
        'Anchors'              : layers.Anchors,
        'ClipBoxes'            : layers.ClipBoxes,
        'RegressBoxes'         : layers.RegressBoxes,
        'FilterDetections'     : layers.FilterDetections,
        'detection_focal_loss' : losses.make_detection_focal_loss(),
        'detection_huber_loss' : losses.make_detection_huber_loss()
    }
    custom_objects.update(load_backbone_custom_objects(backbone))

    return keras.models.load_model(filepath, custom_objects=custom_objects)


def RetinaNetTrain(num_classes, **kwargs):
    """ Construct a RetinaNet model for training

    Args
        Refer to keras_collections.models._retinanet_config.py
    Returns
        training_model : RetinaNet training model (a keras.models.Model object)
            - Outputs of this model are [anchor_regressions, anchor_classifications]
            - Shapes would be [(batch_size, num_anchors, 4), (batch_size, num_anchors, num_classes)]
        config         : The network configs (used to convert into a prediction model)
    """
    kwargs['num_classes'] = num_classes
    config = make_config(**kwargs)

    # Make all submodels
    backbone_model = load_backbone(
        config.backbone,
        freeze_backbone=config.freeze_backbone
    )
    regression_model = default_regression_model(
        config.num_anchors,
        pyramid_feature_size=config.pyramid_feature_size,
        regression_feature_size=config.regression_feature_size
    )
    classification_model = default_classification_model(
        config.num_classes, config.num_anchors,
        pyramid_feature_size=config.pyramid_feature_size,
        classification_feature_size=config.classification_feature_size
    )

    # Create inputs and apply preprocessing
    model_inputs = keras.Input(shape=(None, None, 3))
    preprocessing_layer = load_backbone_preprocessing(config.backbone)
    preprocessed_inputs = preprocessing_layer(model_inputs)

    # Create feature pyramid
    C3, C4, C5 = backbone_model(preprocessed_inputs)[-3:]
    features = create_pyramid_features(C3, C4, C5, feature_size=config.pyramid_feature_size)

    # Compute outputs
    regression_outputs     = apply_model_to_features('regression'    , regression_model    , features)
    classification_outputs = apply_model_to_features('classification', classification_model, features)

    return keras.models.Model(
        inputs=model_inputs,
        outputs=(regression_outputs, classification_outputs),
        name=config.name + '_train'
    ), config


def RetinaNetFromTrain(
    training_model,
    config,
    nms=True,
    class_specific_filter=True,
):
    """ Converts a RetinaNet model for training from a prediction model

    Args
        training_model        : The RetinaNetTrain mode
        config                : The configs returned by the training model
        nms                   : Flag to trigger if nms is to be applied
        class_specific_filter : Flag to trigger if nms is to be applied to each class
    Returns
        RetinaNet prediction model (a keras.models.Model object)
            - Outputs of this model are [boxes, scores, labels]
            - Shapes would be [(batch_size, max_detection, 4), (batch_size, max_detection), (batch_size, max_detection)]
    """
    # Compute anchors
    features = [training_model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors  = compute_anchors(
        features,
        sizes=config.anchor_sizes,
        strides=config.anchor_strides,
        ratios=config.anchor_ratios,
        scales=config.anchor_scales,
    )

    # Get training_model outputs
    regression     = training_model.outputs[0]
    classification = training_model.outputs[1]

    # Apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([training_model.inputs[0], boxes])

    # Filter detections
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification])

    return keras.models.Model(inputs=training_model.inputs, outputs=detections, name=config.name)
