import keras
from .. import backend


class ClipBoxes(keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())

        x1 = backend.clip_by_value(boxes[:, :, 0], 0, shape[2])
        y1 = backend.clip_by_value(boxes[:, :, 1], 0, shape[1])
        x2 = backend.clip_by_value(boxes[:, :, 2], 0, shape[2])
        y2 = backend.clip_by_value(boxes[:, :, 3], 0, shape[1])

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
