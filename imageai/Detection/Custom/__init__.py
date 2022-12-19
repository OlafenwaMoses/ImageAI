import os
import time
import math
import json
import warnings
from typing import List, Union, Tuple, Dict
from collections import defaultdict

import numpy as np
from PIL import Image
import cv2
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

from .yolo.dataset import LoadImagesAndLabels
from .yolo.custom_anchors import generate_anchors
from .yolo.compute_loss import compute_loss
from .yolo import validate
from ...yolov3.tiny_yolov3 import YoloV3Tiny
from ...yolov3.yolov3 import YoloV3
from ...yolov3.utils import draw_bbox_and_label, get_predictions, prepare_image 


class DetectionModelTrainer:
    """
    This is the Detection Model training class, which allows you to train object detection models
    on image datasets that are in YOLO format, using the YOLOv3.
    """

    def __init__(self) -> None:
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__cuda = (self.__device != "cpu")
        self.__model_type = ""
        self.__model = None
        self.__optimizer = None
        self.__data_dir = ""
        self.__classes: List[str] = None
        self.__num_classes = None
        self.__anchors = None
        self.__dataset_name = None
        self.__mini_batch_size: int = None
        self.__scaler = amp.GradScaler(enabled=self.__cuda)
        self.__lr_lambda = None
        self.__custom_train_dataset = None
        self.__custom_val_dataset = None
        self.__train_loader = None
        self.__val_loader = None

        self.__model_path: str = None
        self.__epochs: int = None
        self.__output_models_dir: str = None
        self.__output_json_dir: str = None

    def __set_training_param(self, epochs : int, accumulate : int) -> None:
        self.__lr_lambda = lambda x : ((1 - math.cos(x * math.pi / epochs)) / 2  ) * (0.1 - 1.0) + 1.0
        self.__anchors = generate_anchors(
                                self.__custom_train_dataset,
                                n=9 if self.__model_type=="yolov3" else 6
                            )
        self.__anchors = [round(i) for i in self.__anchors.reshape(-1).tolist()]
        if self.__model_type == "yolov3":
            self.__model = YoloV3(
                        num_classes=self.__num_classes,
                        anchors=self.__anchors,
                        device=self.__device
                    )
        elif self.__model_type == "tiny-yolov3":
            self.__model = YoloV3Tiny(
                        num_classes=self.__num_classes,
                        anchors=self.__anchors,
                        device=self.__device
                    )
        if self.__model_path:
            self.__load_model()

        w_d = (5e-4) * (self.__mini_batch_size * accumulate / 64) # scale weight decay
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for m in self.__model.modules():
            if hasattr(m, 'bias') and isinstance(m.bias, torch.nn.Parameter):  # bias
                g2.append(m.bias)
            if isinstance(m, torch.nn.BatchNorm2d):  # weight (no decay)
                g0.append(m.weight)
            elif hasattr(m, 'weight') and isinstance(m.weight, torch.nn.Parameter):  # weight (with decay)
                g1.append(m.weight)

        self.__optimizer = SGD(
                    g0,
                    lr=1e-4,
                    momentum=0.937,
                    weight_decay=w_d,
                    nesterov=True
                )
        self.__optimizer.add_param_group({'params': g1, 'weight_decay': w_d})  # add g1 with weight_decay
        self.__optimizer.add_param_group({'params': g2})  # add g2 (biases)
        self.__lr_scheduler = lr_scheduler.LambdaLR(
                                self.__optimizer,
                                lr_lambda=self.__lr_lambda
                            )
        del g0, g1, g2
        self.__model.to(self.__device)

    def __load_model(self) -> None:
        try:
            state_dict = torch.load(self.__model_path, map_location=self.__device)
            # check against cases where number of classes differs, causing the
            # channel of the convolutional layer just before the detection layer
            # to differ.
            new_state_dict = {k:v for k,v in state_dict.items() if k in self.__model.state_dict().keys() and v.shape==self.__model.state_dict()[k].shape}
            self.__model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print("pretrained weight loading failed. Defaulting to using random weight.")
        
        print("="*20)
        print("Pretrained YOLOv3 model loaded to initialize weights")
        print("="*20)

    def __load_data(self) -> None:
        self.__num_classes = len(self.__classes)
        self.__dataset_name = os.path.basename(os.path.dirname(self.__data_dir+os.path.sep))
        self.__custom_train_dataset = LoadImagesAndLabels(self.__data_dir, train=True)
        self.__custom_val_dataset = LoadImagesAndLabels(self.__data_dir, train=False)
        self.__train_loader = DataLoader(
                            self.__custom_train_dataset, batch_size=self.__mini_batch_size,
                            shuffle=True,
                            collate_fn=self.__custom_train_dataset.collate_fn
                        )
        self.__val_loader = DataLoader(
                            self.__custom_val_dataset, batch_size=self.__mini_batch_size//2,
                            shuffle=True, collate_fn=self.__custom_val_dataset.collate_fn
                        )

    def setModelTypeAsYOLOv3(self) -> None:
        self.__model_type = "yolov3"

    def setModelTypeAsTinyYOLOv3(self) -> None:
        self.__model_type = "tiny-yolov3"

    def setDataDirectory(self, data_directory: str):
        if os.path.isdir(data_directory):
            self.__data_dir = data_directory
        else:
            raise ValueError(
                    "The parameter passed should point to a valid directory"
                )
    def setTrainConfig(self, object_names_array: List[str], batch_size: int=4, num_experiments=100, train_from_pretrained_model: str = None):
        self.__model_path = train_from_pretrained_model
        self.__classes = object_names_array
        self.__mini_batch_size = batch_size
        self.__epochs = num_experiments
        self.__output_models_dir = os.path.join(self.__data_dir, "models")
        self.__output_json_dir = os.path.join(self.__data_dir, "json")

    def trainModel(self) -> None:

        self.__load_data()
        os.makedirs(self.__output_models_dir, exist_ok=True)
        os.makedirs(self.__output_json_dir, exist_ok=True)

        mp, mr, map50, map50_95, best_fitness = 0, 0, 0, 0, 0.0
        nbs = 64 # norminal batch size
        nb = len(self.__train_loader) # number of batches
        nw = max(3 * nb, 1000)  # number of warmup iterations.
        last_opt_step = -1
        prev_save_name, recent_save_name = "", ""

        accumulate = max(round(nbs / self.__mini_batch_size), 1) # accumulate loss before optimizing.

        self.__set_training_param(self.__epochs, accumulate)

        with open(os.path.join(self.__output_json_dir, f"{self.__dataset_name}_{self.__model_type}_detection_config.json"), "w") as configWriter:
            json.dump(
                {
                    "labels": self.__classes,
                    "anchors": self.__anchors
                },
                configWriter
            )

        since = time.time()

        self.__lr_scheduler.last_epoch = -1

        for epoch in range(1, self.__epochs+1):
            self.__optimizer.zero_grad()
            mloss = torch.zeros(3, device=self.__device)
            print(f"Epoch {epoch}/{self.__epochs}", "-"*10, sep="\n")

            for phase in ["train", "validation"]:
                if phase=="train":
                    self.__model.train()
                    print("Train: ")
                    for batch_i, (data, anns) in enumerate(self.__train_loader):
                        batches_done = batch_i + nb * epoch

                        data = data.to(self.__device)
                        anns = anns.to(self.__device)

                        # warmup
                        if batches_done <= nw:
                            xi = [0, nw]  # x interp
                            accumulate = max(1, np.interp(batches_done, xi, [1, nbs / self.__mini_batch_size]).round())
                            for j, x in enumerate(self.__optimizer.param_groups):
                                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                                x['lr'] = np.interp(batches_done, xi, [0.1 if j == 2 else 0.0, 0.01 * self.__lr_lambda(epoch)])
                                if 'momentum' in x:
                                    x['momentum'] = np.interp(batches_done, xi, [0.8, 0.9])

                        with amp.autocast(enabled=self.__cuda):
                            _ = self.__model(data)
                            loss_layers = self.__model.get_loss_layers()
                            loss, loss_components = compute_loss(loss_layers, anns.detach(), self.__device)

                        self.__scaler.scale(loss).backward()
                        mloss = (mloss * batch_i + loss_components) / (batch_i + 1)

                       # Optimize
                        if batches_done - last_opt_step >= accumulate:
                            self.__scaler.step(self.__optimizer)  # optimizer.step
                            self.__scaler.update()
                            self.__optimizer.zero_grad()
                            last_opt_step = batches_done

                    print(f"    box loss-> {float(mloss[0]):.5f}, object loss-> {float(mloss[1]):.5f}, class loss-> {float(mloss[2]):.5f}")

                    self.__lr_scheduler.step()

                else:
                    self.__model.eval()
                    print("Validation:")

                    mp, mr, map50, map50_95 = validate.run(
                                                self.__model, self.__val_loader,
                                                self.__num_classes, device=self.__device
                                            )
                    
                    print(f"    recall: {mr:0.6f} precision: {mp:0.6f} mAP@0.5: {map50:0.6f}, mAP@0.5-0.95: {map50_95:0.6f}" "\n")

                    if map50 > best_fitness:
                        best_fitness = map50
                        recent_save_name = self.__model_type+f"_{self.__dataset_name}_mAP-{best_fitness:0.5f}_epoch-{epoch}.pt"
                        if prev_save_name:
                            os.remove(os.path.join(self.__output_models_dir, prev_save_name))
                        torch.save(
                            self.__model.state_dict(),
                            os.path.join(self.__output_models_dir, recent_save_name)
                        )
                        prev_save_name = recent_save_name

            if epoch == self.__epochs:
                torch.save(
                        self.__model.state_dict(),
                        os.path.join(self.__output_models_dir, self.__model_type+f"_{self.__dataset_name}_last.pt")
                    )

        elapsed_time = time.time() - since
        print(f"Training completed in {elapsed_time//60:.0f}m {elapsed_time % 60:.0f}s")
        torch.cuda.empty_cache()


class CustomObjectDetection:
    """
    This is the object detection class for using your custom trained models. 
    It supports your custom trained YOLOv3 and TinyYOLOv3 model and allows 
    to you to perform object detection in images.
    """
    def __init__(self) -> None:
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__anchors: List[int] = None
        self.__classes: List[str] = None 
        self.__model = None
        self.__model_loaded: bool = False
        self.__model_path: str = None
        self.__json_path: str = None
        self.__model_type: str = None
        self.__nms_score = 0.4
        self.__objectness_score = 0.4
    
    def setModelTypeAsYOLOv3(self) -> None:
        self.__model_type = "yolov3"

    def setModelTypeAsTinyYOLOv3(self) -> None:
        self.__model_type = "tiny-yolov3"
    
    def setModelPath(self, detection_model_path: str):
        if os.path.isfile(detection_model_path):
            self.__model_path = detection_model_path
            self.__model_loaded = False
        else:
            raise ValueError(
                        "invalid path, path not pointing to the weightfile."
                    ) from None
        self.__model_path = detection_model_path
    
    def setJsonPath(self,  configuration_json: str):
        self.__json_path = configuration_json
    
    def __load_classes_and_anchors(self) -> List[str]:

        with open(self.__json_path) as f:
            json_config = json.load(f)
            self.__anchors = json_config["anchors"]
            self.__classes = json_config["labels"]

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
    
    def loadModel(self) -> None:
        """
        Loads the pretrained weights in the specified model path.
        """
        self.__load_classes_and_anchors()

        if self.__model_type == "yolov3":
            self.__model = YoloV3(
                anchors=self.__anchors,
                num_classes=len(self.__classes),
                device=self.__device
            )
        elif self.__model_type == "tiny-yolov3":
            self.__model = YoloV3Tiny(
                anchors=self.__anchors,
                num_classes=len(self.__classes),
                device=self.__device
            )
        else:
            raise ValueError(f"Invalid model type. Call setModelTypeAsYOLOv3() or setModelTypeAsTinyYOLOv3() to set a model type before loading the model")
                            
        self.__model.to(self.__device)

        state_dict = torch.load(self.__model_path, map_location=self.__device)
        try:
            self.__model.load_state_dict(state_dict)
            self.__model_loaded = True
            self.__model.to(self.__device).eval()
        except Exception as e:
            raise RuntimeError(f"Invalid weights!!! {e}")


    def detectObjectsFromImage(self,
                input_image: Union[str, np.ndarray, Image.Image],
                output_image_path: str=None,
                output_type: str ="file",
                extract_detected_objects: bool=False, minimum_percentage_probability: int=40,
                display_percentage_probability: bool=True, display_object_name: bool=True,
                display_box: bool=True,
                custom_objects: List=None,
                nms_treshold: float= 0.4,
                objectness_treshold: float= 0.4,
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
        
        self.__nms_score = nms_treshold
        self.__objectness_score = objectness_treshold
        
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