"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

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
from ..utils.image import preprocess_image

from . import retinanet
from . import Backbone


class MobileNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    allowed_backbones = ['mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224']

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return mobilenet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Download pre-trained weights for the specified backbone name.
        This name is in the format mobilenet{rows}_{alpha} where rows is the
        imagenet shape dimension and 'alpha' controls the width of the network.
        For more info check the explanation from the keras mobilenet script itself.
        """

        alpha = float(self.backbone.split('_')[1])
        rows = int(self.backbone.split('_')[0].replace('mobilenet', ''))

        # load weights
        if keras.backend.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        model_name = 'mobilenet_{}_{}_tf_no_top.h5'.format(alpha_text, rows)
        weights_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/' + model_name
        weights_path = keras.utils.get_file(model_name, weights_url, cache_subdir='models')

        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        backbone = self.backbone.split('_')[0]

        if backbone not in MobileNetBackbone.allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, MobileNetBackbone.allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')


def mobilenet_retinanet(num_classes, backbone='mobilenet224_1.0', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a mobilenet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a MobileNet backbone.
    """
    alpha = float(backbone.split('_')[1])

    # choose default input
    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    backbone = keras.applications.mobilenet.MobileNet(input_tensor=inputs, alpha=alpha, include_top=False, pooling=None, weights=None)

    # create the full model
    layer_names = ['conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu']
    layer_outputs = [backbone.get_layer(name).output for name in layer_names]
    backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)

    # invoke modifier if given
    if modifier:
        backbone = modifier(backbone)

    # C2 not provided
    backbone_layers = {
        'C3': backbone.outputs[0],
        'C4': backbone.outputs[1],
        'C5': backbone.outputs[2]
    }

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone_layers, **kwargs)
