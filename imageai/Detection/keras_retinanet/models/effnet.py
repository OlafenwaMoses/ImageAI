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

from tensorflow import keras

from . import retinanet
from . import Backbone
import efficientnet.keras as efn


class EfficientNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(EfficientNetBackbone, self).__init__(backbone)
        self.preprocess_image_func = None

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return effnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        from efficientnet.weights import IMAGENET_WEIGHTS_PATH
        from efficientnet.weights import IMAGENET_WEIGHTS_HASHES

        model_name = 'efficientnet-b' + self.backbone[-1]
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
        weights_path = keras.utils.get_file(file_name, IMAGENET_WEIGHTS_PATH + file_name, cache_subdir='models', file_hash=file_hash)
        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                             'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return efn.preprocess_input(inputs)


def effnet_retinanet(num_classes, backbone='EfficientNetB0', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            # inputs = keras.layers.Input(shape=(224, 224, 3))
            inputs = keras.layers.Input(shape=(None, None, 3))

    # get last conv layer from the end of each block [28x28, 14x14, 7x7]
    if backbone == 'EfficientNetB0':
        model = efn.EfficientNetB0(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB1':
        model = efn.EfficientNetB1(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB2':
        model = efn.EfficientNetB2(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB3':
        model = efn.EfficientNetB3(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB4':
        model = efn.EfficientNetB4(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB5':
        model = efn.EfficientNetB5(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB6':
        model = efn.EfficientNetB6(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'EfficientNetB7':
        model = efn.EfficientNetB7(input_tensor=inputs, include_top=False, weights=None)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    layer_outputs = ['block4a_expand_activation', 'block6a_expand_activation', 'top_activation']

    layer_outputs = [
        model.get_layer(name=layer_outputs[0]).output,  # 28x28
        model.get_layer(name=layer_outputs[1]).output,  # 14x14
        model.get_layer(name=layer_outputs[2]).output,  # 7x7
    ]
    # create the densenet backbone
    model = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=model.name)

    # invoke modifier if given
    if modifier:
        model = modifier(model)

    # C2 not provided
    backbone_layers = {
        'C3': model.outputs[0],
        'C4': model.outputs[1],
        'C5': model.outputs[2]
    }

    # create the full model
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone_layers, **kwargs)


def EfficientNetB0_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB0', inputs=inputs, **kwargs)


def EfficientNetB1_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB1', inputs=inputs, **kwargs)


def EfficientNetB2_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB2', inputs=inputs, **kwargs)


def EfficientNetB3_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB3', inputs=inputs, **kwargs)


def EfficientNetB4_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB4', inputs=inputs, **kwargs)


def EfficientNetB5_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB5', inputs=inputs, **kwargs)


def EfficientNetB6_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB6', inputs=inputs, **kwargs)


def EfficientNetB7_retinanet(num_classes, inputs=None, **kwargs):
    return effnet_retinanet(num_classes=num_classes, backbone='EfficientNetB7', inputs=inputs, **kwargs)
