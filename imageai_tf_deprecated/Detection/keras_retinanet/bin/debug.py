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
import cv2

# Set keycodes for changing images
# 81, 83 are left and right arrows on linux in Ascii code (probably not needed)
# 65361, 65363 are left and right arrows in linux
# 2424832, 2555904 are left and right arrows on Windows
# 110, 109 are 'n' and 'm' on mac, windows, linux
# (unfortunately arrow keys not picked up on mac)
leftkeys = (81, 110, 65361, 2424832)
rightkeys = (83, 109, 65363, 2555904)

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..utils.anchors import anchors_for_shape, compute_gt_annotations
from ..utils.config import read_config_file, parse_anchor_parameters, parse_pyramid_levels
from ..utils.image import random_visual_effect_generator
from ..utils.tf_version import check_tf_version
from ..utils.transform import random_transform_generator
from ..utils.visualization import draw_annotations, draw_boxes, draw_caption


def create_generator(args):
    """ Create the data generators.

    Args:
        args: parseargs arguments object.
    """
    common_args = {
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'group_method'     : args.group_method
    }

    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )

    visual_effect_generator = random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05)
    )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        generator = CocoGenerator(
            args.coco_path,
            args.coco_set,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        generator = PascalVocGenerator(
            args.pascal_path,
            args.pascal_set,
            image_extension=args.image_extension,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )
    elif args.dataset_type == 'csv':
        generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )
    elif args.dataset_type == 'oid':
        generator = OpenImagesGenerator(
            args.main_dir,
            subset=args.subset,
            version=args.version,
            labels_filter=args.labels_filter,
            parent_label=args.parent_label,
            annotation_cache_dir=args.annotation_cache_dir,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        generator = KittiGenerator(
            args.kitti_path,
            subset=args.subset,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path',  help='Path to dataset directory (ie. /tmp/COCO).')
    coco_parser.add_argument('--coco-set', help='Name of the set to show (defaults to val2017).', default='val2017')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--pascal-set',  help='Name of the set to show (defaults to test).', default='test')
    pascal_parser.add_argument('--image-extension',   help='Declares the dataset images\' extension.', default='.jpg')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')
    kitti_parser.add_argument('subset', help='Argument for loading a subset from train/val.')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('subset', help='Argument for loading a subset from train/validation/test.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes',     help='Path to a CSV file containing class label mapping.')

    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors', help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--display-name', help='Display image name on the bottom left corner.', action='store_true')
    parser.add_argument('--show-annotations', help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')
    parser.add_argument('--no-gui', help='Do not open a GUI window. Save images to an output directory instead.', action='store_true')
    parser.add_argument('--output-dir', help='The output directory to save images to if --no-gui is specified.', default='.')
    parser.add_argument('--flatten-output', help='Flatten the folder structure of saved output images into a single folder.', action='store_true')
    parser.add_argument('--group-method', help='Determines how images are grouped together', type=str, default='ratio', choices=['none', 'random', 'ratio'])

    return parser.parse_args(args)


def run(generator, args, anchor_params, pyramid_levels):
    """ Main loop.

    Args
        generator: The generator to debug.
        args: parseargs args object.
    """
    # display images, one at a time
    i = 0
    while True:
        # load the data
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)
        if len(annotations['labels']) > 0 :
            # apply random transformations
            if args.random_transform:
                image, annotations = generator.random_transform_group_entry(image, annotations)
                image, annotations = generator.random_visual_effect_group_entry(image, annotations)

            # resize the image and annotations
            if args.resize:
                image, image_scale = generator.resize_image(image)
                annotations['bboxes'] *= image_scale

            anchors = anchors_for_shape(image.shape, anchor_params=anchor_params, pyramid_levels=pyramid_levels)
            positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

            # draw anchors on the image
            if args.anchors:
                draw_boxes(image, anchors[positive_indices], (255, 255, 0), thickness=1)

            # draw annotations on the image
            if args.show_annotations:
                # draw annotations in red
                draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)

                # draw regressed anchors in green to override most red annotations
                # result is that annotations without anchors are red, with anchors are green
                draw_boxes(image, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0))

            # display name on the image
            if args.display_name:
                draw_caption(image, [0, image.shape[0]], os.path.basename(generator.image_path(i)))

        # write to file and advance if no-gui selected
        if args.no_gui:
            output_path = make_output_path(args.output_dir, generator.image_path(i), flatten=args.flatten_output)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            i += 1
            if i == generator.size():  # have written all images
                break
            else:
                continue

        # if we are using the GUI, then show an image
        cv2.imshow('Image', image)
        key = cv2.waitKeyEx()

        # press right for next image and left for previous (linux or windows, doesn't work for macOS)
        # if you run macOS, press "n" or "m" (will also work on linux and windows)

        if key in rightkeys:
            i = (i + 1) % generator.size()
        if key in leftkeys:
            i -= 1
            if i < 0:
                i = generator.size() - 1

        # press q or Esc to quit
        if (key == ord('q')) or (key == 27):
            return False

    return True


def make_output_path(output_dir, image_path, flatten = False):
    """ Compute the output path for a debug image. """

    # If the output hierarchy is flattened to a single folder, throw away all leading folders.
    if flatten:
        path = os.path.basename(image_path)

    # Otherwise, make sure absolute paths are taken relative to the filesystem root.
    else:
        # Make sure to drop drive letters on Windows, otherwise relpath wil fail.
        _, path = os.path.splitdrive(image_path)
        if os.path.isabs(path):
            path = os.path.relpath(path, '/')

    # In all cases, append "_debug" to the filename, before the extension.
    base, extension = os.path.splitext(path)
    path = base + "_debug" + extension

    # Finally, join the whole thing to the output directory.
    return os.path.join(output_dir, path)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure tensorflow is the minimum required version
    check_tf_version()

    # create the generator
    generator = create_generator(args)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    pyramid_levels = None
    if args.config and 'pyramid_levels' in args.config:
        pyramid_levels = parse_pyramid_levels(args.config)
    # create the display window if necessary
    if not args.no_gui:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    run(generator, args, anchor_params=anchor_params, pyramid_levels=pyramid_levels)


if __name__ == '__main__':
    main()
