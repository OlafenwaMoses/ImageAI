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

import configparser
import numpy as np
from tensorflow import keras
from ..utils.anchors import AnchorParameters


def read_config_file(config_path):
    config = configparser.ConfigParser()

    with open(config_path, 'r') as file:
        config.read_file(file)

    assert 'anchor_parameters' in config, \
        "Malformed config file. Verify that it contains the anchor_parameters section."

    config_keys = set(config['anchor_parameters'])
    default_keys = set(AnchorParameters.default.__dict__.keys())

    assert config_keys <= default_keys, \
        "Malformed config file. These keys are not valid: {}".format(config_keys - default_keys)

    if 'pyramid_levels' in config:
        assert('levels' in config['pyramid_levels']), "pyramid levels specified by levels key"

    return config


def parse_anchor_parameters(config):
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))
    assert (len(sizes) == len(strides)), "sizes and strides should have an equal number of values"

    return AnchorParameters(sizes, strides, ratios, scales)


def parse_pyramid_levels(config):
    levels = list(map(int, config['pyramid_levels']['levels'].split(' ')))

    return levels
