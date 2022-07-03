#!/usr/bin/env python

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

import argparse
import os
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..utils.config import read_config_file, parse_anchor_parameters, parse_pyramid_levels
from ..utils.gpu import setup_gpu
from ..utils.tf_version import check_tf_version


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument('model_in', help='The model to convert.')
    parser.add_argument('model_out', help='Path to save the converted model to.')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet50')
    parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
    parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')
    parser.add_argument('--nms-threshold', help='Value for non maximum suppression threshold.', type=float, default=0.5)
    parser.add_argument('--score-threshold', help='Threshold for prefiltering boxes.', type=float, default=0.05)
    parser.add_argument('--max-detections', help='Maximum number of detections to keep.', type=int, default=300)
    parser.add_argument('--parallel-iterations', help='Number of batch items to process in parallel.', type=int, default=32)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure tensorflow is the minimum required version
    check_tf_version()

    # set modified tf session to avoid using the GPUs
    setup_gpu('cpu')

    # optionally load config parameters
    anchor_parameters = None
    pyramid_levels = None
    if args.config:
        args.config = read_config_file(args.config)
        if 'anchor_parameters' in args.config:
            anchor_parameters = parse_anchor_parameters(args.config)

        if 'pyramid_levels' in args.config:
            pyramid_levels = parse_pyramid_levels(args.config)

    # load the model
    model = models.load_model(args.model_in, backbone_name=args.backbone)

    # check if this is indeed a training model
    models.check_training_model(model)

    # convert the model
    model = models.convert_model(
        model,
        nms=args.nms,
        class_specific_filter=args.class_specific_filter,
        anchor_params=anchor_parameters,
        pyramid_levels=pyramid_levels,
        nms_threshold=args.nms_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        parallel_iterations=args.parallel_iterations
    )

    # save model
    model.save(args.model_out)


if __name__ == '__main__':
    main()
