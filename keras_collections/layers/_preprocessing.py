import numpy as np
import keras


class InceptionPreprocess(keras.layers.Layer):
    """Performs preprocessing for a inception backbone"""
    def __init__(self, *args, **kwargs):
        _INCEPTION_SCALE = 1 / 127.5
        _INCEPTION_BIAS  = -1
        self.scale = keras.backend.constant(_INCEPTION_SCALE)
        self.bias  = keras.backend.constant(_INCEPTION_BIAS)
        super(InceptionPreprocess, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        x = inputs * self.scale + self.bias
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class ResNetPreprocess(keras.layers.Layer):
    """Performs preprocessing for a resnet backbone"""
    def __init__(self, *args, **kwargs):
        _RESNET_MEAN = np.array([103.939, 116.779, 123.68], keras.backend.floatx())
        self.bias = keras.backend.constant(-_RESNET_MEAN)
        super(ResNetPreprocess, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        x = inputs[..., ::-1]
        x = keras.backend.bias_add(x, self.bias)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
