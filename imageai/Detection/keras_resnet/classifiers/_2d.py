# -*- coding: utf-8 -*-

"""
keras_resnet.classifiers
~~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular residual two-dimensional classifiers.
"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers
from keras.layers.merge import Concatenate
from imageai.Detection import keras_resnet
from imageai.Detection.keras_resnet import models


class ResNet18(keras.models.Model):
    """
    A :class:`ResNet18 <ResNet18>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> from imageai.Detection.keras_resnet import classifiers
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = classifiers.ResNet18(x, classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet18(inputs, include_top=False)

        # Concatenate partially flatten intermediate heads
        x = Concatenate()([keras.layers.Flatten()(out) for out in outputs.output])

        x = keras.layers.Dense(classes, activation="softmax")(x)

        super(ResNet18, self).__init__(inputs, x)


class ResNet34(keras.models.Model):
    """
    A :class:`ResNet34 <ResNet34>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> from imageai.Detection.keras_resnet import classifiers
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = classifiers.ResNet34(x, classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet34(inputs, include_top=False)

        # Concatenate partially flatten intermediate heads
        x = Concatenate()([keras.layers.Flatten()(out) for out in outputs.output])

        x = keras.layers.Dense(classes, activation="softmax")(x)

        super(ResNet34, self).__init__(inputs, x)


class ResNet50(keras.models.Model):
    """
    A :class:`ResNet50 <ResNet50>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> from imageai.Detection.keras_resnet import classifiers
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = classifiers.ResNet50(x, classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet50(inputs, include_top=False)

        # Concatenate partially flatten intermediate heads
        x = Concatenate()([keras.layers.Flatten()(out) for out in outputs.output])

        x = keras.layers.Dense(classes, activation="softmax")(x)

        super(ResNet50, self).__init__(inputs, x)


class ResNet101(keras.models.Model):
    """
    A :class:`ResNet101 <ResNet101>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> from imageai.Detection.keras_resnet import classifiers
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = classifiers.ResNet101(x, classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet101(inputs, include_top=False)

        # Concatenate partially flatten intermediate heads
        x = Concatenate()([keras.layers.Flatten()(out) for out in outputs.output])

        x = keras.layers.Dense(classes, activation="softmax")(x)

        super(ResNet101, self).__init__(inputs, x)


class ResNet152(keras.models.Model):
    """
    A :class:`ResNet152 <ResNet152>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> from imageai.Detection.keras_resnet import classifiers
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = classifiers.ResNet152(x, classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet152(inputs, include_top=False)

        # Concatenate partially flatten intermediate heads
        x = Concatenate()([keras.layers.Flatten()(out) for out in outputs.output])

        x = keras.layers.Dense(classes, activation="softmax")(x)

        super(ResNet152, self).__init__(inputs, x)


class ResNet200(keras.models.Model):
    """
    A :class:`ResNet200 <ResNet200>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> from imageai.Detection.keras_resnet import classifiers
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = classifiers.ResNet200(x, classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet200(inputs, include_top=False)

        # Concatenate partially flatten intermediate heads
        x = Concatenate()([keras.layers.Flatten()(out) for out in outputs.output])

        x = keras.layers.Dense(classes, activation="softmax")(x)

        super(ResNet200, self).__init__(inputs, x)
