# -*- coding: utf-8 -*-

"""
keras_resnet.models._2d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular two-dimensional residual models.
"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers

from imageai.Detection import keras_resnet as keras_resnet

from imageai.Detection.keras_resnet import blocks
from imageai.Detection.keras_resnet import layers


def ResNet(inputs, blocks, block, include_top=True, classes=1000, freeze_bn=True, numerical_names=None, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_2d

        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1")(inputs)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    x = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0

        x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
        x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

        return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)


def ResNet18(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet18 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet18(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [2, 2, 2, 2]

    return ResNet(inputs, blocks, block=keras_resnet.blocks.basic_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet34(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet34 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet34(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]

    return ResNet(inputs, blocks, block=keras_resnet.blocks.basic_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet50 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet50(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]
    numerical_names = [False, False, False, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=keras_resnet.blocks.bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet101(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet101 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet101(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 23, 3]
    numerical_names = [False, True, True, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=keras_resnet.blocks.bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet152(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet152 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet152(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 8, 36, 3]
    numerical_names = [False, True, True, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=keras_resnet.blocks.bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet200(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet200 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet200(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 24, 36, 3]
    numerical_names = [False, True, True, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=keras_resnet.blocks.bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)
