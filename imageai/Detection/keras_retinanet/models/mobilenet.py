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

import keras
from keras.applications.mobilenet import MobileNet, BASE_WEIGHT_PATH, get_file, relu6, DepthwiseConv2D

from ..models import retinanet

mobile_net_custom_objects = {
    'relu6': relu6,
    'DepthwiseConv2D': DepthwiseConv2D
}

custom_objects = retinanet.custom_objects.copy()
custom_objects.update(mobile_net_custom_objects)

allowed_backbones = ['mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224']


def download_imagenet(backbone):
    """ Download pre-trained weights for the specified backbone name. This name is in the format
        mobilenet{rows}_{alpha} where rows is the imagenet shape dimension and 'alpha' controls
        the width of the network.
        For more info check the explanation from the keras mobilenet script itself
    # Arguments
        backbone    : Backbone name.
    """

    alpha = float(backbone.split('_')[1])
    rows = int(backbone.split('_')[0].replace('mobilenet', ''))

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
    weights_url = BASE_WEIGHT_PATH + model_name
    weights_path = get_file(model_name, weights_url, cache_subdir='models')

    return weights_path


def validate_backbone(backbone):
    """ Validate the backbone choice.
    # Arguments
        backbone    : Backbone name.
    """

    backbone = backbone.split('_')[0]

    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))


def mobilenet_retinanet(num_classes, backbone='mobilenet224_1.0', inputs=None, modifier=None, **kwargs):
    alpha = float(backbone.split('_')[1])

    # choose default input
    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    mobilenet = MobileNet(input_tensor=inputs, alpha=alpha, include_top=False, pooling=None, weights=None)

    # get last layer from each depthwise convolution blocks 3, 5, 11 and 13
    outputs = [mobilenet.get_layer(name='conv_pw_{}_relu'.format(i)).output for i in [3, 5, 11, 13]]

    # create the mobilenet backbone
    mobilenet = keras.models.Model(inputs=inputs, outputs=outputs, name=mobilenet.name)

    # invoke modifier if given
    if modifier:
        mobilenet = modifier(mobilenet)

    # create the full model
    model = retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone=mobilenet, **kwargs)

    return model
