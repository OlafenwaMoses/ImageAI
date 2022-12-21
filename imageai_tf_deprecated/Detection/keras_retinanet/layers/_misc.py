"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow
from tensorflow import keras
from .. import backend
from ..utils import anchors as utils_anchors

import numpy as np


class Anchors(keras.layers.Layer):
    """ Keras layer for generating achors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Initializer for an Anchors layer.

        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
            scales: The scales of the anchors to generate (defaults to AnchorParameters.default.scales).
        """
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios  = utils_anchors.AnchorParameters.default.ratios
        elif isinstance(ratios, list):
            self.ratios  = np.array(ratios)
        if scales is None:
            self.scales  = utils_anchors.AnchorParameters.default.scales
        elif isinstance(scales, list):
            self.scales  = np.array(scales)

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.anchors = utils_anchors.generate_anchors(
            base_size=self.size,
            ratios=self.ratios,
            scales=self.scales,
        ).astype(np.float32)

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)

        # generate proposals from bbox deltas and shifted anchors
        if keras.backend.image_data_format() == 'channels_first':
            anchors = backend.shift(features_shape[2:4], self.stride, self.anchors)
        else:
            anchors = backend.shift(features_shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            if keras.backend.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })

        return config


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = tensorflow.transpose(source, (0, 2, 3, 1))
            output = backend.resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = tensorflow.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return backend.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return backend.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class ClipBoxes(keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """
    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
        if keras.backend.image_data_format() == 'channels_first':
            _, _, height, width = tensorflow.unstack(shape, axis=0)
        else:
            _, height, width, _ = tensorflow.unstack(shape, axis=0)

        x1, y1, x2, y2 = tensorflow.unstack(boxes, axis=-1)
        x1 = tensorflow.clip_by_value(x1, 0, width  - 1)
        y1 = tensorflow.clip_by_value(y1, 0, height - 1)
        x2 = tensorflow.clip_by_value(x2, 0, width  - 1)
        y2 = tensorflow.clip_by_value(y2, 0, height - 1)

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
