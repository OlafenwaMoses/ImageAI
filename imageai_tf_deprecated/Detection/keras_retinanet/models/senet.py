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
from classification_models.keras import Classifiers


class SeBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(SeBackbone, self).__init__(backbone)
        _, self.preprocess_image_func = Classifiers.get(self.backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return senet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        from classification_models.weights import WEIGHTS_COLLECTION

        weights_path = None
        for el in WEIGHTS_COLLECTION:
            if el['model'] == self.backbone and not el['include_top']:
                weights_path = keras.utils.get_file(el['name'], el['url'], cache_subdir='models', file_hash=el['md5'])

        if weights_path is None:
            raise ValueError('Unable to find imagenet weights for backbone {}!'.format(self.backbone))

        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
                             'seresnext50', 'seresnext101', 'senet154']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return self.preprocess_image_func(inputs)


def senet_retinanet(num_classes, backbone='seresnext50', inputs=None, modifier=None, **kwargs):
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

    classifier, _ = Classifiers.get(backbone)
    model = classifier(input_tensor=inputs, include_top=False, weights=None)

    # get last conv layer from the end of each block [28x28, 14x14, 7x7]
    if backbone == 'seresnet18' or backbone == 'seresnet34':
        layer_outputs = ['stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1']
    elif backbone == 'seresnet50':
        layer_outputs = ['activation_36', 'activation_66', 'activation_81']
    elif backbone == 'seresnet101':
        layer_outputs = ['activation_36', 'activation_151', 'activation_166']
    elif backbone == 'seresnet152':
        layer_outputs = ['activation_56', 'activation_236', 'activation_251']
    elif backbone == 'seresnext50':
        layer_outputs = ['activation_37', 'activation_67', 'activation_81']
    elif backbone == 'seresnext101':
        layer_outputs = ['activation_37', 'activation_152', 'activation_166']
    elif backbone == 'senet154':
        layer_outputs = ['activation_59', 'activation_239', 'activation_253']
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

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


def seresnet18_retinanet(num_classes, inputs=None, **kwargs):
    return senet_retinanet(num_classes=num_classes, backbone='seresnet18', inputs=inputs, **kwargs)


def seresnet34_retinanet(num_classes, inputs=None, **kwargs):
    return senet_retinanet(num_classes=num_classes, backbone='seresnet34', inputs=inputs, **kwargs)


def seresnet50_retinanet(num_classes, inputs=None, **kwargs):
    return senet_retinanet(num_classes=num_classes, backbone='seresnet50', inputs=inputs, **kwargs)


def seresnet101_retinanet(num_classes, inputs=None, **kwargs):
    return senet_retinanet(num_classes=num_classes, backbone='seresnet101', inputs=inputs, **kwargs)


def seresnet152_retinanet(num_classes, inputs=None, **kwargs):
    return senet_retinanet(num_classes=num_classes, backbone='seresnet152', inputs=inputs, **kwargs)


def seresnext50_retinanet(num_classes, inputs=None, **kwargs):
    return senet_retinanet(num_classes=num_classes, backbone='seresnext50', inputs=inputs, **kwargs)


def seresnext101_retinanet(num_classes, inputs=None, **kwargs):
    return senet_retinanet(num_classes=num_classes, backbone='seresnext101', inputs=inputs, **kwargs)


def senet154_retinanet(num_classes, inputs=None, **kwargs):
    return senet_retinanet(num_classes=num_classes, backbone='senet154', inputs=inputs, **kwargs)
