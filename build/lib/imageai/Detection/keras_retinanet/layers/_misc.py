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

import keras
from .. import backend
from ..utils import anchors as utils_anchors

import numpy as np


class Anchors(keras.layers.Layer):
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        elif isinstance(ratios, list):
            self.ratios  = np.array(ratios)
        if scales is None:
            self.scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
        elif isinstance(scales, list):
            self.scales  = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors     = keras.backend.variable(utils_anchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)[:3]

        # generate proposals from bbox deltas and shifted anchors
        anchors = backend.shift(features_shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
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


class NonMaximumSuppression(keras.layers.Layer):
    def __init__(self, nms_threshold=0.5, score_threshold=0.05, max_boxes=300, *args, **kwargs):
        self.nms_threshold   = nms_threshold
        self.score_threshold = score_threshold
        self.max_boxes       = max_boxes
        super(NonMaximumSuppression, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        # TODO: support batch size > 1.
        boxes           = inputs[0][0]
        classification  = inputs[1][0]
        other           = [i[0] for i in inputs[2:]]  # can be any user-specified additional data
        indices         = backend.range(keras.backend.shape(classification)[0])
        selected_scores = []

        # perform per class NMS
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]

            # threshold based on score
            score_indices = backend.where(keras.backend.greater(scores, self.score_threshold))
            score_indices = keras.backend.cast(score_indices, 'int32')
            boxes_        = backend.gather_nd(boxes, score_indices)
            scores        = keras.backend.gather(scores, score_indices)[:, 0]

            # perform NMS
            nms_indices = backend.non_max_suppression(boxes_, scores, max_output_size=self.max_boxes, iou_threshold=self.nms_threshold)

            # filter set of original indices
            selected_indices = keras.backend.gather(score_indices, nms_indices)

            # mask original classification column, setting all suppressed values to 0
            scores = keras.backend.gather(scores, nms_indices)
            scores = backend.scatter_nd(selected_indices, scores, keras.backend.shape(classification[:, c]))
            scores = keras.backend.expand_dims(scores, axis=1)

            selected_scores.append(scores)

        # reconstruct the (suppressed) classification scores
        classification = keras.backend.concatenate(selected_scores, axis=1)

        # reconstruct into the expected output
        detections = keras.backend.concatenate([boxes, classification] + other, axis=1)

        return keras.backend.expand_dims(detections, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], sum([i[2] for i in input_shape]))

    def get_config(self):
        config = super(NonMaximumSuppression, self).get_config()
        config.update({
            'nms_threshold'   : self.nms_threshold,
            'score_threshold' : self.score_threshold,
            'max_boxes'       : self.max_boxes,
        })

        return config


class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return backend.resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    def __init__(self, mean=None, std=None, *args, **kwargs):
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.1, 0.1, 0.2, 0.2])

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
