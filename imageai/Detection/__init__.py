import os, warnings
from tkinter import Image
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
from ..retinanet.utils import read_image, draw_bounding_boxes_and_labels, tensor_to_ndarray
import uuid

from ..backend_check.model_extension import extension_check

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

    * setModelPath: Used to specify the filepath to the pretrained model.

    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsRetinaNet(), setModelTypeAsYOLOv3(), setModelTypeAsTinyYOLOv3()]

    * loadModel: [This must be called once only before performing object detection]
    Once the above functions have been called, you can call the detectObjectsFromImage() function of
    the object detection instance object at anytime to obtain observable objects in any image.

    * detectObjectsFromImage: Used to perform object detection on an image
    """
    def __init__(self) -> None:
        self.__device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.__nms_score: float = 0.4
        self.__objectness_score: float = 0.5
        self.__anchors: List[int] = None
        self.__anchors_yolov3: List[int] = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        self.__anchors_tiny_yolov3: List[int] = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
                          
        self.__classes = self.__load_classes(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco_classes.txt"))
        self.__model_type = ""
        self.__model = None
        self.__model_loaded = False
        self.__model_path = ""
    
    def __load_classes(self, path: str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def __load_image_yolo(self, input_image : Union[str, np.ndarray, Image.Image]) -> Tuple[List[str], List[np.ndarray], torch.Tensor, torch.Tensor]:
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
        if type(input_image) == str:
            fnames.append(os.path.basename(input_image)) 
        else:
            fnames.append("") 
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

        temp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{str(uuid.uuid4())}.jpg" 
        ) 
        if type(input_image) == np.ndarray:
            cv2.imwrite(temp_path, input_image)
        elif "PIL" in str(type(input_image)):
            input_image.save(temp_path)
        else:
            raise ValueError(
                f"Invalid image input. Supported formats are OpenCV/Numpy array, PIL image or image file path"
            )

        return temp_path

    def __load_image_retinanet(self, input_image : str) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Loads image from the given path.
        """
        allowed_file_extensions = ["jpg", "jpeg", "png"]
        images = []
        scaled_images = []
        fnames = []
        
        delete_file = False
        if type(input_image) is not str:
            input_image = self.__save_temp_img(input_image=input_image)
            delete_file = True


        if os.path.isfile(input_image):
            if input_image.rsplit('.')[-1].lower() in allowed_file_extensions:
                img = read_image(input_image, ImageReadMode.RGB)
                images.append(img)
                scaled_images.append(img.div(255.0).to(self.__device))
                fnames.append(os.path.basename(input_image))
        else:
            raise ValueError(f"Input image with path {input_image} not a valid file")

        if delete_file:
            os.remove(input_image)
        
        if images:
            return (fnames, images, scaled_images)
        raise RuntimeError(
                    f"Error loading image from input."
                    "\nEnsure the folder contains images,"
                    " allowed file extensions are .jpg, .jpeg, .png"
                )
    
    def setModelTypeAsYOLOv3(self):
        """
        'setModelTypeAsYOLOv3()' is used to set the model type to the YOLOv3 model.
        :return:
        """
        self.__anchors = self.__anchors_yolov3
        self.__model_type = "yolov3"
    
    def setModelTypeAsTinyYOLOv3(self):
        """
        'setModelTypeAsTinyYOLOv3()' is used to set the model type to the TinyYOLOv3 model.
        :return:
        """
        self.__anchors = self.__anchors_tiny_yolov3
        self.__model_type = "tiny-yolov3"
    
    def setModelTypeAsRetinaNet(self):
        """
        'setModelTypeAsRetinaNet()' is used to set the model type to the RetinaNet model.
        :return:
        """
        self.__anchors = self.__anchors_tiny_yolov3
        self.__model_type = "retinanet"

    def setModelPath(self, path: str) -> None:
        """
        'setModelPath()' function is required and is used to set the file path to the model adopted from the list of the
        available 3 model types. The model path must correspond to the model type.
        :param model_path:
        :return:
        """
        if os.path.isfile(path):
            extension_check(path)
            self.__model_path = path
            self.__model_loaded = False
        else:
            raise ValueError(
                        "invalid path, path not pointing to a valid file."
                    ) from None
    
    def useCPU(self):
        """
        Used to force classification to be done on CPU.
        By default, classification will occur on GPU compute if available else CPU compute.
        """

        self.__device = "cpu"
        if self.__model_loaded:
            self.__model_loaded = False
            self.loadModel()
    
    def loadModel(self) -> None:
        """
        'loadModel()' function is used to load the model weights into the model architecture from the file path defined
        in the setModelPath() function.
        :return:
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
                raise ValueError(f"Invalid model type. Call setModelTypeAsYOLOv3(), setModelTypeAsTinyYOLOv3() or setModelTypeAsRetinaNet to set a model type before loading the model")

            state_dict = torch.load(self.__model_path, map_location=self.__device)
            try:
                self.__model.load_state_dict(state_dict)
                self.__model_loaded = True
                self.__model.to(self.__device).eval()
            except:
                raise RuntimeError("Invalid weights!!!") from None
    
    def CustomObjects(self, **kwargs):

        """
        The 'CustomObjects()' function allows you to handpick the type of objects ( from the COCO classes ) you want to detect
        from an image. The objects are pre-initiated in the function variables and predefined as 'False',
        which you can easily set to true for any number of objects available.  This function
        returns a dictionary which must be parsed into the 'detectObjectsFromImage()'. Detecting
        custom objects only happens when you call the function 'detectObjectsFromImage()'

        Acceptable values are 'True' and 'False'  for all object values present
        :param boolean_values:
        :return: custom_objects_dict
        """

        if not self.__model_loaded:
            self.loadModel()
        all_objects_str = (obj_label.replace(" ", "_") for obj_label in self.__classes)
        all_objects_dict = {}
        for object_str in all_objects_str:
            all_objects_dict[object_str] = False
        
        for karg in kwargs:
            if karg in all_objects_dict:
                all_objects_dict[karg] = kwargs[karg]
            else:
                raise ValueError(f" object '{karg}' doesn't exist in the supported object classes")

        return all_objects_dict

        

    def detectObjectsFromImage(self,
                input_image: Union[str, np.ndarray, Image.Image],
                output_image_path: str=None,
                output_type: str ="file",
                extract_detected_objects: bool=False, minimum_percentage_probability: int=50,
                display_percentage_probability: bool=True, display_object_name: bool=True,
                display_box: bool=True,
                custom_objects: List=None
               ) -> Union[List[List[Tuple[str, float, Dict[str, int]]]], np.ndarray, List[np.ndarray], List[str]]:
        """
        Detects objects in an image using the unique classes provided
        by COCO.

        :param input_image: path to an image file, cv2 image or PIL image
        :param output_image_path: path to save input image with predictions rendered
        :param output_type: type of output for rendered image. Acceptable values are 'file' and 'array` ( a cv2 image )
        :param extract_detected_objects: extract each object based on the output type
        :param minimum_percentage_probability: the minimum confidence a detected object must have
        :param display_percentage_probability: to diplay/not display the confidence on rendered image   
        :param display_object_name: to diplay/not display the object name on rendered image  
        :param display_box: to diplay/not display the object bounding box on rendered image 
        :param custom_objects: a dictionary of detectable objects set to boolean values
        
        :returns: A list of tuples containing the label of detected object and the
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
                raise RuntimeError(
                        "Model path isn't set, pretrained weights aren't used."
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
            
            if output is None:
                if output_type == "array":
                    if extract_detected_objects:
                        return original_imgs[0], [], []
                    else:
                        return original_imgs[0], []
                else:
                    if extract_detected_objects:
                        return original_imgs[0], []
                    else:
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
                pred_label = self.__classes[int(pred[-1])]
                if custom_objects:
                    if pred_label.replace(" ", "_") in custom_objects.keys():
                        if not custom_objects[pred_label.replace(" ", "_")]:
                            continue
                    else:
                        continue
                predictions[int(pred[0])].append((
                        pred_label,
                        float(pred[-2]),
                        {k:v for k,v in zip(["x1", "y1", "x2", "y2"], map(int, pred[1:5]))},
                    ))
        elif self.__model_type == "retinanet":
            fnames, original_imgs, scaled_images = self.__load_image_retinanet(input_image)
            with torch.no_grad():
                output = self.__model(scaled_images)
            
            if output is None:
                if output_type == "array":
                    if extract_detected_objects:
                        return original_imgs[0], [], []
                    else:
                        return original_imgs[0], []
                else:
                    if extract_detected_objects:
                        return original_imgs[0], []
                    else:
                        return []

            for idx, pred in enumerate(output):
                for id in range(pred["labels"].shape[0]):
                    if pred["scores"][id] >= self.__objectness_score:
                        pred_label = self.__classes[pred["labels"][id]]

                        if custom_objects:
                            if pred_label.replace(" ", "_") in custom_objects.keys():
                                if not custom_objects[pred_label.replace(" ", "_")]:
                                    continue
                            else:
                                continue

                        predictions[idx].append(
                                (
                                    pred_label,
                                    pred["scores"][id].item(),
                                    {k:v for k,v in zip(["x1", "y1", "x2", "y2"], map(int, pred["boxes"][id]))}
                                )
                            )
        
        # Render detection on copy of input image
        original_input_image = None
        output_image_array = None
        extracted_objects = []

        if self.__model_type == "yolov3" or self.__model_type == "tiny-yolov3":
            original_input_image = cv2.cvtColor(original_imgs[0], cv2.COLOR_RGB2BGR)
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
                
                output_image_array = cv2.cvtColor(original_imgs[0], cv2.COLOR_RGB2BGR)
                
        elif self.__model_type == "retinanet":
            original_input_image = tensor_to_ndarray(original_imgs[0].div(255.0))
            original_input_image = cv2.cvtColor(original_input_image, cv2.COLOR_RGB2BGR)
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
                
            output_image_array = tensor_to_ndarray(original_imgs[0].div(255.0))
            output_image_array = cv2.cvtColor(output_image_array, cv2.COLOR_RGB2BGR)
        

        # Format predictions for function reponse
        predictions_batch = list(predictions.values())
        predictions_list = predictions_batch[0] if len(predictions_batch) > 0 else []
        min_probability = minimum_percentage_probability / 100


        if output_type == "file":
            if output_image_path:
                cv2.imwrite(output_image_path, output_image_array)

                if extract_detected_objects:
                    extraction_dir = ".".join(output_image_path.split(".")[:-1]) + "-extracted"
                    os.mkdir(extraction_dir)
                    count = 0
                    for obj_prediction in predictions_list: 
                        if obj_prediction[1] >= min_probability:
                            count += 1
                            extracted_path = os.path.join(
                                extraction_dir, 
                                ".".join(os.path.basename(output_image_path).split(".")[:-1]) + f"-{count}.jpg"
                            )
                            obj_bbox = obj_prediction[2]
                            cv2.imwrite(extracted_path, original_input_image[obj_bbox["y1"] : obj_bbox["y2"], obj_bbox["x1"] : obj_bbox["x2"]])

                            extracted_objects.append(extracted_path)

        elif output_type == "array":
            if extract_detected_objects:
                for obj_prediction in predictions_list: 
                    if obj_prediction[1] >= min_probability:
                        obj_bbox = obj_prediction[2]

                        extracted_objects.append(original_input_image[obj_bbox["y1"] : obj_bbox["y2"], obj_bbox["x1"] : obj_bbox["x2"]])
        else:
            raise ValueError(f"Invalid output_type '{output_type}'. Supported values are 'file' and 'array' ")

        
        predictions_list = [
            {
                "name": prediction[0], "percentage_probability": round(prediction[1] * 100, 2),
                "box_points": [prediction[2]["x1"], prediction[2]["y1"], prediction[2]["x2"], prediction[2]["y2"]]
            } for prediction in predictions_list if prediction[1] >= min_probability
        ]


        if output_type == "array":
            if extract_detected_objects:
                return output_image_array, predictions_list, extracted_objects
            else:
                return output_image_array, predictions_list
        else:
            if extract_detected_objects:
                return predictions_list, extracted_objects
            else:
                return predictions_list


class VideoObjectDetection:
    """
    This is the object detection class for videos and camera live stream inputs in the ImageAI library. It provides support for RetinaNet,
    YOLOv3 and TinyYOLOv3 object detection networks. After instantiating this class, you can set it's properties and
    make object detections using it's pre-defined functions.
    The following functions are required to be called before object detection can be made
    * setModelPath()
    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsRetinaNet(), setModelTypeAsYOLOv3(), setModelTinyYOLOv3()]
    * loadModel() [This must be called once only before performing object detection]
    Once the above functions have been called, you can call the detectObjectsFromVideo() function
    or the detectCustomObjectsFromVideo() of  the object detection instance object at anytime to
    obtain observable objects in any video or camera live stream.
    """

    def __init__(self):
        self.__detector = ObjectDetection()

    def setModelTypeAsYOLOv3(self):
        self.__detector.setModelTypeAsYOLOv3()
    
    def setModelTypeAsTinyYOLOv3(self):
        self.__detector.setModelTypeAsTinyYOLOv3()
    
    def setModelTypeAsRetinaNet(self):
        self.__detector.setModelTypeAsRetinaNet()

    def setModelPath(self, model_path: str):
        extension_check(model_path)
        self.__detector.setModelPath(model_path)

    def loadModel(self):
        self.__detector.loadModel()
    
    def useCPU(self):
        self.__detector.useCPU()
    
    def CustomObjects(self, **kwargs):
        return self.__detector.CustomObjects(**kwargs)

    def detectObjectsFromVideo(self, input_file_path="", camera_input=None, output_file_path="", frames_per_second=20,
                               frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False,
                               display_percentage_probability=True, display_object_name=True, display_box=True, save_detected_video=True,
                               per_frame_function=None, per_second_function=None, per_minute_function=None,
                               video_complete_function=None, return_detected_frame=False, detection_timeout = None, custom_objects=None):

        """
        'detectObjectsFromVideo()' function is used to detect objects observable in the given video path or a camera input:
        * input_file_path , which is the file path to the input video. It is required only if 'camera_input' is not set
        * camera_input , allows you to parse in camera input for live video detections
        * output_file_path , which is the path to the output video. It is required only if 'save_detected_video' is not set to False
        * frames_per_second , which is the number of frames to be used in the output video
        * frame_detection_interval (optional, 1 by default)  , which is the intervals of frames that will be detected.
        * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
        * log_progress (optional) , which states if the progress of the frame processed is to be logged to console
        * display_percentage_probability (optional), can be used to hide or show probability scores on the detected video frames
        * display_object_name (optional), can be used to show or hide object names on the detected video frames
        * save_save_detected_video (optional, True by default), can be set to or not to save the detected video
        * per_frame_function (optional), this parameter allows you to parse in a function you will want to execute after each frame of the video is detected. If this parameter is set to a function, after every video  frame is detected, the function will be executed with the following values parsed into it:
            -- position number of the frame
            -- an array of dictinaries, with each dictionary corresponding to each object detected. Each dictionary contains 'name', 'percentage_probability' and 'box_points'
            -- a dictionary with with keys being the name of each unique objects and value are the number of instances of the object present
            -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed as the fourth value into the function
        * per_second_function (optional), this parameter allows you to parse in a function you will want to execute after each second of the video is detected. If this parameter is set to a function, after every second of a video is detected, the function will be executed with the following values parsed into it:
            -- position number of the second
            -- an array of dictionaries whose keys are position number of each frame present in the last second , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame
            -- an array of dictionaries, with each dictionary corresponding to each frame in the past second, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame
            -- a dictionary with its keys being the name of each unique object detected throughout the past second, and the key values are the average number of instances of the object found in all the frames contained in the past second
            -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                                                                as the fifth value into the function
        * per_minute_function (optional), this parameter allows you to parse in a function you will want to execute after each minute of the video is detected. If this parameter is set to a function, after every minute of a video is detected, the function will be executed with the following values parsed into it:
            -- position number of the minute
            -- an array of dictionaries whose keys are position number of each frame present in the last minute , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame
            -- an array of dictionaries, with each dictionary corresponding to each frame in the past minute, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame
            -- a dictionary with its keys being the name of each unique object detected throughout the past minute, and the key values are the average number of instances of the object found in all the frames contained in the past minute
            -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed as the fifth value into the function
        * video_complete_function (optional), this parameter allows you to parse in a function you will want to execute after all of the video frames have been detected. If this parameter is set to a function, after all of frames of a video is detected, the function will be executed with the following values parsed into it:
            -- an array of dictionaries whose keys are position number of each frame present in the entire video , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame
            -- an array of dictionaries, with each dictionary corresponding to each frame in the entire video, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame
            -- a dictionary with its keys being the name of each unique object detected throughout the entire video, and the key values are the average number of instances of the object found in all the frames contained in the entire video
        * return_detected_frame (optionally, False by default), option to obtain the return the last detected video frame into the per_per_frame_function, per_per_second_function or per_per_minute_function
        * detection_timeout (optionally, None by default), option to state the number of seconds of a video that should be detected after which the detection function stop processing the video
        * thread_safe (optional, False by default), enforce the loaded detection model works across all threads if set to true, made possible by forcing all Tensorflow inference to run on the default graph.
                :param input_file_path:
                :param camera_input
                :param output_file_path:
                :param save_detected_video:
                :param frames_per_second:
                :param frame_detection_interval:
                :param minimum_percentage_probability:
                :param log_progress:
                :param display_percentage_probability:
                :param display_object_name:
                :param per_frame_function:
                :param per_second_function:
                :param per_minute_function:
                :param video_complete_function:
                :param return_detected_frame:
                :param detection_timeout:
                :param thread_safe:
                :return output_video_filepath:
                :return counting:
                :return output_objects_array:
                :return output_objects_count:
                :return detected_copy:
                :return this_second_output_object_array:
                :return this_second_counting_array:
                :return this_second_counting:
                :return this_minute_output_object_array:
                :return this_minute_counting_array:
                :return this_minute_counting:
                :return this_video_output_object_array:
                :return this_video_counting_array:
                :return this_video_counting:
        """

        if (input_file_path == "" and camera_input == None):
            raise ValueError(
                "You must set 'input_file_path' to a valid video file, or set 'camera_input' to a valid camera")
        elif (save_detected_video == True and output_file_path == ""):
            raise ValueError(
                "You must set 'output_video_filepath' to a valid video file name, in which the detected video will be saved. If you don't intend to save the detected video, set 'save_detected_video=False'")

        else:
            try:

                output_frames_dict = {}
                output_frames_count_dict = {}

                input_video = cv2.VideoCapture(input_file_path)
                if (camera_input != None):
                    input_video = camera_input

                output_video_filepath = output_file_path + '.mp4'

                frame_width = int(input_video.get(3))
                frame_height = int(input_video.get(4))
                output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc(*"MP4V"),
                                               frames_per_second,
                                               (frame_width, frame_height))

                counting = 0

                detection_timeout_count = 0
                video_frames_count = 0

                while (input_video.isOpened()):
                    ret, frame = input_video.read()

                    if (ret == True):

                        video_frames_count += 1
                        if (detection_timeout != None):
                            if ((video_frames_count % frames_per_second) == 0):
                                detection_timeout_count += 1

                            if (detection_timeout_count >= detection_timeout):
                                break

                        output_objects_array = []

                        counting += 1

                        if (log_progress == True):
                            print("Processing Frame : ", str(counting))

                        detected_copy = frame.copy()

                        check_frame_interval = counting % frame_detection_interval

                        if (counting == 1 or check_frame_interval == 0):
                            try:
                                detected_copy, output_objects_array = self.__detector.detectObjectsFromImage(
                                    input_image=frame, output_type="array",
                                    minimum_percentage_probability=minimum_percentage_probability,
                                    display_percentage_probability=display_percentage_probability,
                                    display_object_name=display_object_name,
                                    display_box=display_box,
                                    custom_objects=custom_objects)
                            except:
                                None

                        output_frames_dict[counting] = output_objects_array

                        output_objects_count = {}
                        for eachItem in output_objects_array:
                            eachItemName = eachItem["name"]
                            try:
                                output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                            except:
                                output_objects_count[eachItemName] = 1

                        output_frames_count_dict[counting] = output_objects_count

                        
                        if (save_detected_video == True):
                            output_video.write(detected_copy)

                        if (counting == 1 or check_frame_interval == 0):
                            if (per_frame_function != None):
                                if (return_detected_frame == True):
                                    per_frame_function(counting, output_objects_array, output_objects_count,
                                                       detected_copy)
                                elif (return_detected_frame == False):
                                    per_frame_function(counting, output_objects_array, output_objects_count)

                        if (per_second_function != None):
                            if (counting != 1 and (counting % frames_per_second) == 0):

                                this_second_output_object_array = []
                                this_second_counting_array = []
                                this_second_counting = {}

                                for aa in range(counting):
                                    if (aa >= (counting - frames_per_second)):
                                        this_second_output_object_array.append(output_frames_dict[aa + 1])
                                        this_second_counting_array.append(output_frames_count_dict[aa + 1])

                                for eachCountingDict in this_second_counting_array:
                                    for eachItem in eachCountingDict:
                                        try:
                                            this_second_counting[eachItem] = this_second_counting[eachItem] + \
                                                                             eachCountingDict[eachItem]
                                        except:
                                            this_second_counting[eachItem] = eachCountingDict[eachItem]

                                for eachCountingItem in this_second_counting:
                                    this_second_counting[eachCountingItem] = int(this_second_counting[eachCountingItem] / frames_per_second)

                                if (return_detected_frame == True):
                                    per_second_function(int(counting / frames_per_second),
                                                        this_second_output_object_array, this_second_counting_array,
                                                        this_second_counting, detected_copy)

                                elif (return_detected_frame == False):
                                    per_second_function(int(counting / frames_per_second),
                                                        this_second_output_object_array, this_second_counting_array,
                                                        this_second_counting)

                        if (per_minute_function != None):

                            if (counting != 1 and (counting % (frames_per_second * 60)) == 0):

                                this_minute_output_object_array = []
                                this_minute_counting_array = []
                                this_minute_counting = {}

                                for aa in range(counting):
                                    if (aa >= (counting - (frames_per_second * 60))):
                                        this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                        this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                                for eachCountingDict in this_minute_counting_array:
                                    for eachItem in eachCountingDict:
                                        try:
                                            this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                                                             eachCountingDict[eachItem]
                                        except:
                                            this_minute_counting[eachItem] = eachCountingDict[eachItem]

                                for eachCountingItem in this_minute_counting:
                                    this_minute_counting[eachCountingItem] = int(this_minute_counting[eachCountingItem] / (frames_per_second * 60))

                                if (return_detected_frame == True):
                                    per_minute_function(int(counting / (frames_per_second * 60)),
                                                        this_minute_output_object_array, this_minute_counting_array,
                                                        this_minute_counting, detected_copy)

                                elif (return_detected_frame == False):
                                    per_minute_function(int(counting / (frames_per_second * 60)),
                                                        this_minute_output_object_array, this_minute_counting_array,
                                                        this_minute_counting)


                    else:
                        break

                if (video_complete_function != None):

                    this_video_output_object_array = []
                    this_video_counting_array = []
                    this_video_counting = {}

                    for aa in range(counting):
                        this_video_output_object_array.append(output_frames_dict[aa + 1])
                        this_video_counting_array.append(output_frames_count_dict[aa + 1])

                    for eachCountingDict in this_video_counting_array:
                        for eachItem in eachCountingDict:
                            try:
                                this_video_counting[eachItem] = this_video_counting[eachItem] + \
                                                                eachCountingDict[eachItem]
                            except:
                                this_video_counting[eachItem] = eachCountingDict[eachItem]

                    for eachCountingItem in this_video_counting:
                        this_video_counting[eachCountingItem] = int(this_video_counting[eachCountingItem] / counting)

                    video_complete_function(this_video_output_object_array, this_video_counting_array,
                                            this_video_counting)

                input_video.release()
                output_video.release()

                if (save_detected_video == True):
                    return output_video_filepath

            except:
                raise ValueError(
                    "An error occured. It may be that your input video is invalid. Ensure you specified a proper string value for 'output_file_path' is 'save_detected_video' is not False. "
                    "Also ensure your per_frame, per_second, per_minute or video_complete_analysis function is properly configured to receive the right parameters. ")