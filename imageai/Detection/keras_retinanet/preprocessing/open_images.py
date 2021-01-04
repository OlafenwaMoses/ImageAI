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


def load_hierarchy(metadata_dir, version='v4'):
    hierarchy = None
    if version == 'challenge2018':
        hierarchy = 'bbox_labels_500_hierarchy.json'
    elif version == 'v4':
        hierarchy = 'bbox_labels_600_hierarchy.json'
    elif version == 'v3':
        hierarchy = 'bbox_labels_600_hierarchy.json'

    hierarchy_json = os.path.join(metadata_dir, hierarchy)
    with open(hierarchy_json) as f:
        hierarchy_data = json.loads(f.read())

    return hierarchy_data


def load_hierarchy_children(hierarchy):
    res = [hierarchy['LabelName']]

    if 'Subcategory' in hierarchy:
        for subcategory in hierarchy['Subcategory']:
            children = load_hierarchy_children(subcategory)

            for c in children:
                res.append(c)

    return res


def find_hierarchy_parent(hierarchy, parent_cls):
    if hierarchy['LabelName'] == parent_cls:
        return hierarchy
    elif 'Subcategory' in hierarchy:
        for child in hierarchy['Subcategory']:
            res = find_hierarchy_parent(child, parent_cls)
            if res is not None:
                return res

    return None


def get_labels(metadata_dir, version='v4'):
    if version == 'v4' or version == 'challenge2018':
        csv_file = 'class-descriptions-boxable.csv' if version == 'v4' else 'challenge-2018-class-descriptions-500.csv'

        boxable_classes_descriptions = os.path.join(metadata_dir, csv_file)
        id_to_labels = {}
        cls_index    = {}

        i = 0
        with open(boxable_classes_descriptions) as f:
            for row in csv.reader(f):
                # make sure the csv row is not empty (usually the last one)
                if len(row):
                    label       = row[0]
                    description = row[1].replace("\"", "").replace("'", "").replace('`', '')

                    id_to_labels[i]  = description
                    cls_index[label] = i

                    i += 1
    else:
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


def generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index, version='v4'):
    validation_image_ids = {}

    if version == 'v4':
        annotations_path = os.path.join(metadata_dir, subset, '{}-annotations-bbox.csv'.format(subset))
    elif version == 'challenge2018':
        validation_image_ids_path = os.path.join(metadata_dir, 'challenge-2018-image-ids-valset-od.csv')

        with open(validation_image_ids_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=['ImageID'])
            next(reader)
            for line, row in enumerate(reader):
                image_id = row['ImageID']
                validation_image_ids[image_id] = True

        annotations_path = os.path.join(metadata_dir, 'challenge-2018-train-annotations-bbox.csv')
    else:
        annotations_path = os.path.join(metadata_dir, subset, 'annotations-human-bbox.csv')

    fieldnames = ['ImageID', 'Source', 'LabelName', 'Confidence',
                  'XMin', 'XMax', 'YMin', 'YMax',
                  'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']

    id_annotations = dict()
    with open(annotations_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=fieldnames)
        next(reader)

        images_sizes = {}
        for line, row in enumerate(reader):
            frame = row['ImageID']

            if version == 'challenge2018':
                if subset == 'train':
                    if frame in validation_image_ids:
                        continue
                elif subset == 'validation':
                    if frame not in validation_image_ids:
                        continue
                else:
                    raise NotImplementedError('This generator handles only the train and validation subsets')

            class_name = row['LabelName']

            if class_name not in cls_index:
                continue

            cls_id = cls_index[class_name]

            if version == 'challenge2018':
                # We recommend participants to use the provided subset of the training set as a validation set.
                # This is preferable over using the V4 val/test sets, as the training set is more densely annotated.
                img_path = os.path.join(main_dir, 'images', 'train', frame + '.jpg')
            else:
                img_path = os.path.join(main_dir, 'images', subset, frame + '.jpg')

            if frame in images_sizes:
                width, height = images_sizes[frame]
            else:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.width, img.height
                        images_sizes[frame] = (width, height)
                except Exception as ex:
                    if version == 'challenge2018':
                        raise ex
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
            self, main_dir, subset, version='v4',
            labels_filter=None, annotation_cache_dir='.',
            parent_label=None,
            **kwargs
    ):
        if version == 'challenge2018':
            metadata = 'challenge2018'
        elif version == 'v4':
            metadata = '2018_04'
        elif version == 'v3':
            metadata = '2017_11'
        else:
            raise NotImplementedError('There is currently no implementation for versions older than v3')

        if version == 'challenge2018':
            self.base_dir     = os.path.join(main_dir, 'images', 'train')
        else:
            self.base_dir     = os.path.join(main_dir, 'images', subset)

        metadata_dir          = os.path.join(main_dir, metadata)
        annotation_cache_json = os.path.join(annotation_cache_dir, subset + '.json')

        self.hierarchy          = load_hierarchy(metadata_dir, version=version)
        id_to_labels, cls_index = get_labels(metadata_dir, version=version)

        if os.path.exists(annotation_cache_json):
            with open(annotation_cache_json, 'r') as f:
                self.annotations = json.loads(f.read())
        else:
            self.annotations = generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index, version=version)
            json.dump(self.annotations, open(annotation_cache_json, "w"))

        if labels_filter is not None or parent_label is not None:
            self.id_to_labels, self.annotations = self.__filter_data(id_to_labels, cls_index, labels_filter, parent_label)
        else:
            self.id_to_labels = id_to_labels

        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.annotations)])

        super(OpenImagesGenerator, self).__init__(**kwargs)

    def __filter_data(self, id_to_labels, cls_index, labels_filter=None, parent_label=None):
        """
        If you want to work with a subset of the labels just set a list with trainable labels
        :param labels_filter: Ex: labels_filter = ['Helmet', 'Hat', 'Analog television']
        :param parent_label: If parent_label is set this will bring you the parent label
        but also its children in the semantic hierarchy as defined in OID, ex: Animal
        hierarchical tree
        :return:
        """

        children_id_to_labels = {}

        if parent_label is None:
            # there is/are no other sublabel(s) other than the labels itself

            for label in labels_filter:
                for i, lb in id_to_labels.items():
                    if lb == label:
                        children_id_to_labels[i] = label
                        break
        else:
            parent_cls = None
            for i, lb in id_to_labels.items():
                if lb == parent_label:
                    parent_id = i
                    for c, index in cls_index.items():
                        if index == parent_id:
                            parent_cls = c
                    break

            if parent_cls is None:
                raise Exception('Couldnt find label {}'.format(parent_label))

            parent_tree = find_hierarchy_parent(self.hierarchy, parent_cls)

            if parent_tree is None:
                raise Exception('Couldnt find parent {} in the semantic hierarchical tree'.format(parent_label))

            children = load_hierarchy_children(parent_tree)

            for cls in children:
                index = cls_index[cls]
                label = id_to_labels[index]
                children_id_to_labels[index] = label

        id_map = dict([(ind, i) for i, ind in enumerate(children_id_to_labels.keys())])

        filtered_annotations = {}
        for k in self.annotations:
            img_ann = self.annotations[k]

            filtered_boxes = []
            for ann in img_ann['boxes']:
                cls_id = ann['cls_id']
                if cls_id in children_id_to_labels:
                    ann['cls_id'] = id_map[cls_id]
                    filtered_boxes.append(ann)

            if len(filtered_boxes) > 0:
                filtered_annotations[k] = {'w': img_ann['w'], 'h': img_ann['h'], 'boxes': filtered_boxes}

        children_id_to_labels = dict([(id_map[i], l) for (i, l) in children_id_to_labels.items()])

        return children_id_to_labels, filtered_annotations

    def size(self):
        return len(self.annotations)

    def num_classes(self):
        return len(self.id_to_labels)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.id_to_labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError()

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

        annotations = {'labels': np.empty((len(labels),)), 'bboxes': np.empty((len(labels), 4))}
        for idx, ann in enumerate(labels):
            cls_id = ann['cls_id']
            x1 = ann['x1'] * width
            x2 = ann['x2'] * width
            y1 = ann['y1'] * height
            y2 = ann['y2'] * height

            annotations['bboxes'][idx, 0] = x1
            annotations['bboxes'][idx, 1] = y1
            annotations['bboxes'][idx, 2] = x2
            annotations['bboxes'][idx, 3] = y2
            annotations['labels'][idx] = cls_id

        return annotations
