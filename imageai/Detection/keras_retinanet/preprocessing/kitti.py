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
import os.path

import numpy as np
from PIL import Image

from .generator import Generator
from ..utils.image import read_image_bgr

kitti_classes = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 7
}


class KittiGenerator(Generator):
    def __init__(
        self,
        base_dir,
        subset='train',
        **kwargs
    ):
        self.base_dir = base_dir

        label_dir = os.path.join(self.base_dir, subset, 'labels')
        image_dir = os.path.join(self.base_dir, subset, 'images')

        """
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """

        self.id_to_labels = {}
        for label, id in kitti_classes.items():
            self.id_to_labels[id] = label

        self.image_data = dict()
        self.images = []
        for i, fn in enumerate(os.listdir(label_dir)):
            label_fp = os.path.join(label_dir, fn)
            image_fp = os.path.join(image_dir, fn.replace('.txt', '.png'))

            self.images.append(image_fp)

            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'left', 'top', 'right', 'bottom', 'dh', 'dw', 'dl',
                          'lx', 'ly', 'lz', 'ry']
            with open(label_fp, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
                boxes = []
                for line, row in enumerate(reader):
                    label = row['type']
                    cls_id = kitti_classes[label]

                    annotation = {'cls_id': cls_id, 'x1': row['left'], 'x2': row['right'], 'y2': row['bottom'], 'y1': row['top']}
                    boxes.append(annotation)

                self.image_data[i] = boxes

        super(KittiGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.images)

    def num_classes(self):
        return max(kitti_classes.values()) + 1

    def name_to_label(self, name):
        raise NotImplementedError()

    def label_to_name(self, label):
        return self.id_to_labels[label]

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.images[image_index])
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        return read_image_bgr(self.images[image_index])

    def load_annotations(self, image_index):
        annotations = self.image_data[image_index]

        boxes = np.zeros((len(annotations), 5))
        for idx, ann in enumerate(annotations):
            boxes[idx, 0] = float(ann['x1'])
            boxes[idx, 1] = float(ann['y1'])
            boxes[idx, 2] = float(ann['x2'])
            boxes[idx, 3] = float(ann['y2'])
            boxes[idx, 4] = int(ann['cls_id'])
        return boxes
