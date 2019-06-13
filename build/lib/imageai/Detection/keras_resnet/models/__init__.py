# -*- coding: utf-8 -*-

"""
keras_resnet.models
~~~~~~~~~~~~~~~~~~~

This module implements popular residual models.
"""

from ._2d import (
    ResNet,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet200
)

from ._time_distributed_2d import (
    TimeDistributedResNet,
    TimeDistributedResNet18,
    TimeDistributedResNet34,
    TimeDistributedResNet50,
    TimeDistributedResNet101,
    TimeDistributedResNet152,
    TimeDistributedResNet200
)
