import os
import warnings
from typing import Tuple, List

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ....yolov3.utils import prepare_image

class LoadImagesAndLabels(Dataset):

    def __init__(self, path : str, net_dim=(416, 416), train=True):
        if not os.path.isdir(path):
            raise NotADirectoryError("path is not a valid directory!!!")

        super().__init__()

        if train:
            path = os.path.join(path, "train")
        else:
            path = os.path.join(path, "validation")

        self.__net_width, self.__net_height = net_dim
        self.__images_paths = []
        self.shapes = []
        self.labels = []
        for img in os.listdir(os.path.join(path, "images")):
            p = os.path.join(path, "images", img)
            image = cv.imread(p)
            if isinstance(image, np.ndarray):
                l_p = self.__img_path2label_path(p)
                self.__images_paths.append(p)
                self.shapes.append((image.shape[1], image.shape[0]))
                self.labels.append(self.__load_raw_label(l_p))

        self.__nsamples = len(self.__images_paths)
        self.shapes = np.array(self.shapes)

    def __len__(self) -> int:
        return self.__nsamples

    def __img_path2label_path(self, path : str) -> str:
        im, lb = os.sep+"images"+os.sep, os.sep+"annotations"+os.sep
        return lb.join(path.rsplit(im, 1)).rsplit(".", 1)[0] + ".txt"

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.__nsamples:
            raise IndexError("Index out of range.")
        image_path = self.__images_paths[idx]
        label = self.labels[idx].copy()
        image, label = self.__load_data(image_path, label)
        return image, label

    def __xywhn2xyxy(self, nlabel : torch.Tensor, width : int, height : int) -> torch.Tensor:
        """
        Transformed label from normalized center_x, center_y, width, height to
        x_1, y_1, x_2, y_2
        """
        label = nlabel.clone()
        label[:, 1] = (nlabel[:, 1] - (nlabel[:, 3] / 2)) * width
        label[:, 2] = (nlabel[:, 2] - (nlabel[:, 4] / 2)) * height
        label[:, 3] = (nlabel[:, 1] + (nlabel[:, 3] / 2)) * width
        label[:, 4] = (nlabel[:, 2] + (nlabel[:, 4] / 2)) * height

        return label

    def __load_data(self, img_path : str, label : np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img = cv.imread(img_path)
        img_h, img_w = img.shape[:2]
        img = prepare_image(img[:, :, :3], [self.__net_width, self.__net_height])
        lab = self.__process_label(label, img_w, img_h)
        return img.squeeze(), lab

    def __load_raw_label(self, label_path : str):
        if os.path.isfile(label_path):
            with warnings.catch_warnings():
                l = np.loadtxt(label_path).reshape(-1,5)
                assert (l >= 0).all(), "bounding box values should be positive and in range 0 - 1"
                assert (l[:, 1:] <= 1).all(), "bounding box values should be in the range 0 - 1"
        else:
            l = np.zeros((0,5), dtype=np.float32)
        return l

    def __process_label(self, label : np.ndarray, image_width : int, image_height : int) -> torch.Tensor:
        """
        Process corresponding label and resize the ground truth bounding boxes
        to match the dimension of the resizes image.
        """
        #max_box = 50
        scaling_factor = min(
                                self.__net_width/image_width,
                                self.__net_width/image_height
                            )
        #bs = torch.zeros((max_box, 6))
        bs = torch.zeros((len(label), 6))
        if label.size > 0:
            nlabels = torch.from_numpy(label)
            labels = self.__xywhn2xyxy(nlabels, image_width, image_height)
            # scale bounding box to match new image size
            labels[:, [1,3]] = ((labels[:, [1,3]] * scaling_factor) +\
                    (self.__net_width - (image_width * scaling_factor))/2)
            labels[:, [2,4]] = ((labels[:, [2,4]] * scaling_factor) +\
                    (self.__net_width - (image_height * scaling_factor))/2)
            
            # convert x1, y1, x2, y2 to center_x, center_y, width, height
            label_copy = labels.clone()
            labels[:, 1] = (label_copy[:, 3] + label_copy[:, 1])/2
            labels[:, 2] = (label_copy[:, 4] + label_copy[:, 2])/2
            labels[:, 3] = (label_copy[:, 3] - label_copy[:, 1])
            labels[:, 4] = (label_copy[:, 4] - label_copy[:, 2])


            # scale labels by new image dimension
            labels[:, 1:5] /= self.__net_width
            bs[:, 1:] = labels[:, :]
        return bs

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = [data for data in batch if data is not None]
        imgs, bboxes = list(zip(*batch))

        imgs = torch.stack(imgs)

        for i, boxes in enumerate(bboxes):
            boxes[:, 0] = i
        bboxes = torch.cat(bboxes, 0)

        return imgs, bboxes

