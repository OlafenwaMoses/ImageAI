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
from . import backend


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        labels         = y_true
        classification = y_pred

        # compute the divisor: for each image in the batch, we want the number of positive anchors

        # clip the labels to 0, 1 so that we ignore the "ignore" label (-1) in the divisor
        divisor = backend.where(keras.backend.less_equal(labels, 0), keras.backend.zeros_like(labels), labels)
        divisor = keras.backend.max(divisor, axis=2, keepdims=True)
        divisor = keras.backend.cast(divisor, keras.backend.floatx())

        # compute the number of positive anchors
        divisor = keras.backend.sum(divisor, axis=1, keepdims=True)

        #  ensure we do not divide by 0
        divisor = keras.backend.maximum(1.0, divisor)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # normalise by the number of positive anchors for each entry in the minibatch
        cls_loss = cls_loss / divisor

        # filter out "ignore" anchors
        anchor_state = keras.backend.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
        indices      = backend.where(keras.backend.not_equal(anchor_state, -1))

        cls_loss = backend.gather_nd(cls_loss, indices)

        # divide by the size of the minibatch
        return keras.backend.sum(cls_loss) / keras.backend.cast(keras.backend.shape(labels)[0], keras.backend.floatx())

    return _focal


def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, 4]

        # compute the divisor: for each image in the batch, we want the number of positive anchors
        divisor = backend.where(keras.backend.equal(anchor_state, 1), keras.backend.ones_like(anchor_state), keras.backend.zeros_like(anchor_state))
        divisor = keras.backend.sum(divisor, axis=1, keepdims=True)
        divisor = keras.backend.maximum(1.0, divisor)

        # pad the tensor to have shape (batch_size, 1, 1) for future division
        divisor   = keras.backend.expand_dims(divisor, axis=2)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # normalise by the number of positive and negative anchors for each entry in the minibatch
        regression_loss = regression_loss / divisor

        # filter out "ignore" anchors
        indices         = backend.where(keras.backend.equal(anchor_state, 1))
        regression_loss = backend.gather_nd(regression_loss, indices)

        # divide by the size of the minibatch
        regression_loss = keras.backend.sum(regression_loss) / keras.backend.cast(keras.backend.shape(y_true)[0], keras.backend.floatx())

        return regression_loss

    return _smooth_l1
