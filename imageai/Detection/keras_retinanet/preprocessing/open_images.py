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

import csv
import json
import os
import warnings

import numpy as np
from PIL import Image

from .generator import Generator
from ..utils.image import read_image_bgr


def get_labels(metadata_dir):
    trainable_classes_path = os.path.join(metadata_dir, 'classes-bbox-trainable.txt')
    description_path = os.path.join(metadata_dir, 'class-descriptions.csv')

    description_table = {}
    with open(description_path) as f:
        for row in csv.reader(f):
            # make sure the csv row is not empty (usually the last one)
            if len(row):
                description_table[row[0]] = row[1].replace("\"", "").replace("'", "").replace('`', '')

    with open(trainable_classes_path, 'rb') as f:
        trainable_classes = f.read().split('\n')

    id_to_labels = dict([(i, description_table[c]) for i, c in enumerate(trainable_classes)])
    cls_index = dict([(c, i) for i, c in enumerate(trainable_classes)])

    return id_to_labels, cls_index


def generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index):
    annotations_path = os.path.join(metadata_dir, subset, 'annotations-human-bbox.csv')

    cnt = 0
    with open(annotations_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file,
                                fieldnames=['ImageID', 'Source', 'LabelName',
                                            'Confidence', 'XMin', 'XMax', 'YMin',
                                            'YMax'])
        reader.next()
        for _ in reader:
            cnt += 1

    id_annotations = dict()
    with open(annotations_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file,
                                fieldnames=['ImageID', 'Source', 'LabelName',
                                            'Confidence', 'XMin', 'XMax', 'YMin',
                                            'YMax'])
        reader.next()

        images_sizes = {}
        for line, row in enumerate(reader):
            frame = row['ImageID']
            class_name = row['LabelName']

            if class_name not in cls_index:
                continue

            cls_id = cls_index[class_name]

            img_path = os.path.join(main_dir, 'images', subset, frame + '.jpg')
            if frame in images_sizes:
                width, height = images_sizes[frame]
            else:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.width, img.height
                        images_sizes[frame] = (width, height)
                except Exception:
                    continue

            x1 = float(row['XMin'])
            x2 = float(row['XMax'])
            y1 = float(row['YMin'])
            y2 = float(row['YMax'])

            x1_int = int(round(x1 * width))
            x2_int = int(round(x2 * width))
            y1_int = int(round(y1 * height))
            y2_int = int(round(y2 * height))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            if y2_int == y1_int:
                warnings.warn('filtering line {}: rounding y2 ({}) and y1 ({}) makes them equal'.format(line, y2, y1))
                continue

            if x2_int == x1_int:
                warnings.warn('filtering line {}: rounding x2 ({}) and x1 ({}) makes them equal'.format(line, x2, x1))
                continue

            img_id = row['ImageID']
            annotation = {'cls_id': cls_id, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

            if img_id in id_annotations:
                annotations = id_annotations[img_id]
                annotations['boxes'].append(annotation)
            else:
                id_annotations[img_id] = {'w': width, 'h': height, 'boxes': [annotation]}
    return id_annotations


class OpenImagesGenerator(Generator):
    def __init__(
            self, main_dir, subset, version='2017_11',
            labels_filter=None, annotation_cache_dir='.',
            fixed_labels=False,
            **kwargs
    ):
        self.base_dir = os.path.join(main_dir, 'images', subset)
        metadata_dir = os.path.join(main_dir, version)
        annotation_cache_json = os.path.join(annotation_cache_dir, subset + '.json')

        self.id_to_labels, cls_index = get_labels(metadata_dir)

        if os.path.exists(annotation_cache_json):
            with open(annotation_cache_json, 'r') as f:
                self.annotations = json.loads(f.read())
        else:
            self.annotations = generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index)
            json.dump(self.annotations, open(annotation_cache_json, "w"))

        if labels_filter is not None:
            self.id_to_labels, self.annotations = self.__filter_data(labels_filter, fixed_labels)

        self.id_to_image_id = dict()
        for i, k in enumerate(self.annotations):
            self.id_to_image_id[i] = k

        super(OpenImagesGenerator, self).__init__(**kwargs)

    def __filter_data(self, labels_filter, fixed_labels):
        """
        If you want to work with a subset of the labels just set a list with trainable labels
        :param labels_filter: Ex: labels_filter = ['Helmet', 'Hat', 'Analog television']
        :param fixed_labels: If fixed_labels is true this will bring you the 'Helmet' label
        but also: 'bicycle helmet', 'welding helmet', 'ski helmet' etc...
        :return:
        """

        labels_to_id = dict([(l, i) for i, l in enumerate(labels_filter)])

        sub_labels_to_id = {}
        if fixed_labels:
            # there is/are no other sublabel(s) other than the labels itself
            sub_labels_to_id = labels_to_id
        else:
            for l in labels_filter:
                label = str.lower(l)
                for v in [v for v in self.id_to_labels.values() if label in str.lower(v)]:
                    sub_labels_to_id[v] = labels_to_id[l]

        filtered_annotations = {}
        for k in self.annotations:
            img_ann = self.annotations[k]

            filtered_boxes = []
            for ann in img_ann['boxes']:
                cls_id = ann['cls_id']
                label = self.id_to_labels[cls_id]
                if label in sub_labels_to_id:
                    ann['cls_id'] = sub_labels_to_id[label]
                    filtered_boxes.append(ann)

            if len(filtered_boxes) > 0:
                filtered_annotations[k] = {'w': img_ann['w'], 'h': img_ann['h'], 'boxes': filtered_boxes}

        id_to_labels = dict([(labels_to_id[k], k) for k in labels_to_id])
        return id_to_labels, filtered_annotations

    def size(self):
        return len(self.annotations)

    def num_classes(self):
        return len(self.id_to_labels)

    def name_to_label(self, name):
        raise NotImplementedError()

    def label_to_name(self, label):
        return self.id_to_labels[label]

    def image_aspect_ratio(self, image_index):
        img_annotations = self.annotations[self.id_to_image_id[image_index]]
        height, width = img_annotations['h'], img_annotations['w']
        return float(width) / float(height)

    def image_path(self, image_index):
        path = os.path.join(self.base_dir, self.id_to_image_id[image_index] + '.jpg')
        return path

    def load_image(self, image_index):
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        image_annotations = self.annotations[self.id_to_image_id[image_index]]

        labels = image_annotations['boxes']
        height, width = image_annotations['h'], image_annotations['w']

        boxes = np.zeros((len(labels), 5))
        for idx, ann in enumerate(labels):
            cls_id = ann['cls_id']
            x1 = ann['x1'] * width
            x2 = ann['x2'] * width
            y1 = ann['y1'] * height
            y2 = ann['y2'] * height

            boxes[idx, 0] = x1
            boxes[idx, 1] = y1
            boxes[idx, 2] = x2
            boxes[idx, 3] = y2
            boxes[idx, 4] = cls_id

        return boxes
