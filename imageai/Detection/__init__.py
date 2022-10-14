import os, warnings
from tkinter import Image
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from PIL import Image
import torchvision

import numpy as np
from enum import Enum
import torch
import cv2
from typing import Union, List

from ..yolov3.yolov3 import YoloV3
from ..yolov3.tiny_yolov3 import YoloV3Tiny
from ..yolov3.utils import draw_bbox_and_label, get_predictions, prepare_image
from ..retinanet.utils import read_image, draw_bounding_boxes_and_labels, save_image
import torchvision.transforms as transforms
import uuid

warnings.filterwarnings("once", category=ResourceWarning)


class ImageReadMode(Enum):
    """
    Support for various modes while reading images.

    Use ``ImageReadMode.UNCHANGED`` for loading the image as-is,
    ``ImageReadMode.GRAY`` for converting to grayscale,
    ``ImageReadMode.GRAY_ALPHA`` for grayscale with transparency,
    ``ImageReadMode.RGB`` for RGB and ``ImageReadMode.RGB_ALPHA`` for
    RGB with transparency.
    """

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4

class ObjectDetection:
    """
    This is the object detection class for images in the ImageAI library. It allows you to detect the 80 objects in the COCO dataset [ https://cocodataset.org/#home ] in any image. 
    
    This class provides support for RetinaNet, YOLOv3 and TinyYOLOv3 object detection networks . After instantiating this class, you can set its properties and make object detections using pretrained models.

    The following functions are required to be called before object detection can be made

    * setModelPath(): Used to specify the filepath to the pretrained model.

    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsRetinaNet(), setModelTypeAsYOLOv3(), setModelTypeAsTinyYOLOv3()]

    * loadModel(): [This must be called once only before performing object detection]
    Once the above functions have been called, you can call the detectObjectsFromImage() function of
    the object detection instance object at anytime to obtain observable objects in any image.

    * detectObjectsFromImage(): Used to perform object detection on an image
    """
    def __init__(
            self,
            label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco_classes.txt"),
            device : str ="cpu",
            nms_score : float = 0.4,
            objectness_score : float = 0.5,
            ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                    "specified device is cuda, but cuda is unavailable, defaulting to cpu",
                    ResourceWarning
                )
            device="cpu"
        self.__device = device
        self.__nms_score = nms_score
        self.__objectness_score = objectness_score
        self.__anchors = None
        self.__anchors_yolov3 = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        self.__anchors_tiny_yolov3 = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
                          
        self.__classes = self.__load_classes(label_path)
        self.__model_type = ""
        self.__model = None
        self.__model_loaded = False
        self.__model_path = ""
    
    def __load_classes(self, path : str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def __load_image_yolo(self, input_image : Union[str, np.ndarray, Image.Image]) -> Tuple[List[str], List[np.ndarray], torch.Tensor, torch.Tensor]:
        """
        Loads image/images from the given path. If the given path is a directory,
        this function only load the images in the directory (it does noot visit the
        subdirectories).
        """
        allowed_exts = ["jpg", "jpeg", "png"]
        fnames = []
        original_dims = []
        inputs = []
        original_imgs = []
        if type(input_image) == str:
            if os.path.isfile(input_image):
                if input_image.rsplit('.')[-1].lower() in allowed_exts:
                    img = cv2.imread(input_image)
            else:
                raise ValueError(f"image path '{input_image}' is not found or a valid file")
        elif type(input_image) == np.ndarray:
            img = input_image
        elif "PIL" in str(type(input_image)):
            img = np.asarray(input_image)
        else:
            raise ValueError(f"Invalid image input format")
        
        img_h, img_w, _ = img.shape
        original_imgs.append(np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).astype(np.uint8))
        original_dims.append((img_w, img_h))
        fnames.append(os.path.basename(input_image))
        inputs.append(prepare_image(img, (416, 416)))

        if original_dims:
            return (
                    fnames,
                    original_imgs,
                    torch.FloatTensor(original_dims).repeat(1,2).to(self.__device),
                    torch.cat(inputs, 0).to(self.__device)
                    )
        raise RuntimeError(
                    f"Error loading image."
                    "\nEnsure the file is a valid image,"
                    " allowed file extensions are .jpg, .jpeg, .png"
                )
    
    def __save_temp_img(self, input_image : Union[np.ndarray, Image.Image]) -> str:

        temp_path = f"{str(uuid.uuid4())}.jpg" 
        if type(input_image) == np.ndarray:
            cv2.imwrite(temp_path, input_image)
        elif "PIL" in str(type(input_image)):
            input_image.save(temp_path)

        return temp_path

    def __load_image_retinanet(self, image_path : str) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Loads image/images from the given path. If the given path is a directory,
        this function only load the images in the directory (it does noot visit the
        subdirectories).
        """
        allowed_file_extensions = ["jpg", "jpeg", "png"]
        images = []
        scaled_images = []
        fnames = []
        if os.path.isfile(image_path):
            if image_path.rsplit('.')[-1].lower() in allowed_file_extensions:
                img = read_image(image_path, ImageReadMode.RGB)
                images.append(img)
                scaled_images.append(img.div(255.0).to(self.__device))
                fnames.append(os.path.basename(image_path))

        if images:
            return (fnames, images, scaled_images)
        raise RuntimeError(
                    f"Error loading images from {os.path.abspath(image_path)}."
                    "\nEnsure the folder contains images,"
                    " allowed file extensions are .jpg, .jpeg, .png"
                )
    
    def setModelTypeAsYOLOv3(self):
        self.__anchors = self.__anchors_yolov3
        self.__model_type = "yolov3"
    
    def setModelTypeAsTinyYOLOv3(self):
        self.__anchors = self.__anchors_tiny_yolov3
        self.__model_type = "tiny-yolov3"
    
    def setModelTypeAsRetinaNet(self):
        self.__anchors = self.__anchors_tiny_yolov3
        self.__model_type = "retinanet"

    def setModelPath(self, path : str) -> None:
        """
        Sets the path to the pretrained weights.
        """
        if os.path.isfile(path):
            self.__model_path = path
            self.__model_loaded = False
        else:
            raise ValueError(
                        "invalid path, path not pointing to a valid file."
                    ) from None
    
    def loadModel(self) -> None:
        """
        Loads the pretrained weights in the specified model path.
        """
        if not self.__model_loaded:
            if self.__model_type=="yolov3":
                self.__model = YoloV3(
                        anchors=self.__anchors ,
                        num_classes=len(self.__classes),\
                        device=self.__device
                    )
            elif self.__model_type=="tiny-yolov3":
                self.__model = YoloV3Tiny(
                    anchors=self.__anchors,
                    num_classes=len(self.__classes),
                    device=self.__device
                    )
            elif self.__model_type=="retinanet":

                self.__classes = self.__load_classes(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco91_classes.txt"))

                self.__model = torchvision.models.detection.retinanet_resnet50_fpn(
                            pretrained=False, num_classes=91,
                            pretrained_backbone = False
                        )
            else:
                raise ValueError(f"Invalid model type. Call setModelTypeAsYOLOv3() or setModelTypeAsTinyYOLOv3() to set a model type before loading the model")

        
            state_dict = torch.load(self.__model_path, map_location=self.__device)
            try:
                self.__model.load_state_dict(state_dict)
                self.__model_loaded = True
                self.__model.to(self.__device).eval()
            except:
                raise RuntimeError("Invalid weights!!!") from None

    def detectObjectsFromImage(self,
                input_image: Union[str, np.ndarray, Image.Image],
                output_image_path=None,
                output_type="file",
                extract_detected_objects=False, minimum_percentage_probability=50,
                display_percentage_probability=True, display_object_name=True,
                display_box=True,
                custom_objects=None
               ) -> List[List[Tuple[str, float, Dict[str, int]]]]:
        """
        Detects objects in an image using the unique classes provided
        by COCO.

        Parameters:
        -----------
            image_path: path to an image or a directory that contains images.
            verbose:    if true, it prints the output of detection to screen.
            output_dir: directory to place output images with bounding boxes
                        overlayed on the detected object.
        Returns:
        --------
            A list of tuples containing the label of detected object and the
            confidence.
        """
        
        
        self.__model.eval()
        if not self.__model_loaded:
            if self.__model_path:
                warnings.warn(
                        "Model path has changed but pretrained weights in the"
                        " new path is yet to be loaded.",
                        ResourceWarning
                    )
            else:
                warnings.warn(
                        "Model path isn't set, pretrained weights aren't used.",
                        ResourceWarning
                    )
        predictions = defaultdict(lambda : [])
        

        if self.__model_type == "yolov3" or self.__model_type == "tiny-yolov3":
            fnames, original_imgs, input_dims, imgs = self.__load_image_yolo(input_image)
            
            with torch.no_grad():
                output = self.__model(imgs)
            
            output = get_predictions(
                    pred=output.to(self.__device), num_classes=len(self.__classes),
                    nms_confidence_level=self.__nms_score, objectness_confidence= self.__objectness_score,
                    device=self.__device
                )
            
            if isinstance(output, int):
                return []
            # scale the output to match the dimension of the original image
            input_dims = torch.index_select(input_dims, 0, output[:, 0].long())
            scaling_factor = torch.min(416 / input_dims, 1)[0].view(-1, 1)
            output[:, [1,3]] -= (416 - (scaling_factor * input_dims[:, 0].view(-1,1))) / 2
            output[:, [2,4]] -= (416 - (scaling_factor * input_dims[:, 1].view(-1,1))) / 2
            output[:, 1:5] /= scaling_factor

            #clip bounding box for those that extended outside the detected image.
            for idx in range(output.shape[0]):
                output[idx, [1,3]] = torch.clamp(output[idx, [1,3]], 0.0, input_dims[idx, 0])
                output[idx, [2,4]] = torch.clamp(output[idx, [2,4]], 0.0, input_dims[idx, 1])

            for pred in output:
                predictions[int(pred[0])].append((
                        self.__classes[int(pred[-1])],
                        float(pred[-2]),
                        {k:v for k,v in zip(["x1", "y1", "x2", "y2"], map(int, pred[1:5]))},
                    ))
        elif self.__model_type == "retinanet":
            fnames, original_imgs, scaled_images = self.__load_image_retinanet(input_image)
            with torch.no_grad():
                output = self.__model(scaled_images)

            for idx, pred in enumerate(output):
                for id in range(pred["labels"].shape[0]):
                    if pred["scores"][id] >= self.__objectness_score:
                        predictions[idx].append(
                                (
                                    self.__classes[pred["labels"][id]],
                                    pred["scores"][id].item(),
                                    {k:v for k,v in zip(["x1", "y1", "x2", "y2"], map(int, pred["boxes"][id]))}
                                )
                            )

        # TO DO: Unify image + prediction output generation code for supported architectures.
        if output_type == "file":
            if output_image_path:
                if self.__model_type == "yolov3" or self.__model_type == "tiny-yolov3":
                    if isinstance(output, torch.Tensor):
                        for pred in output:
                            percentage_conf = round(float(pred[-2]) * 100, 2)
                            if percentage_conf < minimum_percentage_probability:
                                continue

                            displayed_label = ""
                            if display_object_name:
                                displayed_label = f"{self.__classes[int(pred[-1].item())]} : "
                            if display_percentage_probability:
                                displayed_label += f" {percentage_conf}%"


                            original_imgs[int(pred[0].item())] = draw_bbox_and_label(pred[1:5].int() if display_box else None,
                                displayed_label,
                                original_imgs[int(pred[0].item())]
                            )
                        
                        cv2.imwrite(output_image_path, cv2.cvtColor(original_imgs[0], cv2.COLOR_RGB2BGR))
                elif self.__model_type == "retinanet":
                    for idx, pred in predictions.items():
                        
                        max_dim = max(list(original_imgs[idx].size()))

                        for label, score, bbox in pred:
                            percentage_conf = round(score * 100, 2)
                            if percentage_conf < minimum_percentage_probability:
                                continue
                            
                            displayed_label = ""
                            if display_object_name:
                                displayed_label = f"{label} :"
                            if display_percentage_probability:
                                displayed_label += f" {percentage_conf}%"

                            original_imgs[idx] = draw_bounding_boxes_and_labels(
                                image=original_imgs[idx],
                                boxes=torch.Tensor([[bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]]),
                                draw_boxes=display_box,
                                labels=[displayed_label],
                                label_color=(0, 0, 255),
                                box_color=(0, 255, 0),
                                width=1,
                                fill=False,
                                font_size=int(max_dim / 30)
                            )
                        save_image(original_imgs[idx].div(255.0), output_image_path)

        predictions_list = list(predictions.values())[0]
        min_probability = minimum_percentage_probability / 100
        predictions_list = [
            {
                "name": prediction[0], "percentage_probability": round(prediction[1] * 100, 2),
                "box_points": [prediction[2]["x1"], prediction[2]["y1"], prediction[2]["x2"], prediction[2]["y2"]]
            } for prediction in predictions_list if prediction[1] >= min_probability
        ]

        return predictions_list

