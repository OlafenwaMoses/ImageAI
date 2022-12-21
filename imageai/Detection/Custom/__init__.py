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
from tqdm import tqdm

from .yolo.dataset import LoadImagesAndLabels
from .yolo.custom_anchors import generate_anchors
from .yolo.compute_loss import compute_loss
from .yolo import validate
from ...yolov3.tiny_yolov3 import YoloV3Tiny
from ...yolov3.yolov3 import YoloV3
from ...yolov3.utils import draw_bbox_and_label, get_predictions, prepare_image

from ...backend_check.model_extension import extension_check


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
        # self.__lr_lambda = lambda x : ((1 - math.cos(x * math.pi / epochs)) / 2  ) * (0.1 - 1.0) + 1.0
        self.__lr_lambda = lambda x: (1 - x / (epochs - 1)) * (1.0 - 0.01) + 0.01
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
                    lr=1e-2,
                    momentum=0.6,
                    # weight_decay=w_d,
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
        """
        'setModelTypeAsYOLOv3()' is used to set the model type to the YOLOv3 model.
        :return:
        """
        self.__model_type = "yolov3"

    def setModelTypeAsTinyYOLOv3(self) -> None:
        """
        'setModelTypeAsTinyYOLOv3()' is used to set the model type to the TinyYOLOv3 model.
        :return:
        """
        self.__model_type = "tiny-yolov3"

    def setDataDirectory(self, data_directory: str):
        """
        'setDataDirectory()' is required to set the path to which the data/dataset to be used for training is kept. The input dataset must be in the YOLO format. The directory can have any name, but it must have 'train' and 'validation'
        sub-directory. In the 'train' and 'validation' sub-directories, there must be 'images' and 'annotations'
        sub-directories respectively. The 'images' folder will contain the pictures for the dataset and the
        'annotations' folder will contain the TXT files with details of the annotations for each image in the
        'images folder'.
        N.B: Strictly take note that the filenames (without the extension) of the pictures in the 'images folder'
        must be the same as the filenames (except the extension) of their corresponding annotation TXT files in
        the 'annotations' folder.
        The structure of the 'train' and 'validation' folder must be as follows:
            >> train    >> images       >> img_1.jpg
                        >> images       >> img_2.jpg
                        >> images       >> img_3.jpg
                        >> annotations  >> img_1.txt
                        >> annotations  >> img_2.txt
                        >> annotations  >> img_3.txt
            >> validation   >> images       >> img_151.jpg
                            >> images       >> img_152.jpg
                            >> images       >> img_153.jpg
                            >> annotations  >> img_151.txt
                            >> annotations  >> img_152.txt
                            >> annotations  >> img_153.txt
        :param data_directory:
        :return:
        """
        if os.path.isdir(data_directory):
            self.__data_dir = data_directory
        else:
            raise ValueError(
                    "The parameter passed should point to a valid directory"
                )
    def setTrainConfig(self, object_names_array: List[str], batch_size: int=4, num_experiments=100, train_from_pretrained_model: str = None):
        """
        'setTrainConfig()' function allows you to set the properties for the training instances. It accepts the following values:
        - object_names_array , this is an array of the names of the different objects in your dataset, in the index order your dataset is annotated
        - batch_size (optional),  this is the batch size for the training instance
        - num_experiments (optional),   also known as epochs, it is the number of times the network will train on all the training dataset
        - train_from_pretrained_model (optional), this is used to perform transfer learning by specifying the path to a pre-trained YOLOv3 or TinyYOLOv3 model
        :param object_names_array:
        :param batch_size:
        :param num_experiments:
        :param train_from_pretrained_model:
        :return:
        """
        self.__model_path = train_from_pretrained_model
        if self.__model_path:
            extension_check(self.__model_path)
        self.__classes = object_names_array
        self.__mini_batch_size = batch_size
        self.__epochs = num_experiments
        self.__output_models_dir = os.path.join(self.__data_dir, "models")
        self.__output_json_dir = os.path.join(self.__data_dir, "json")

    def trainModel(self) -> None:
        """
        'trainModel()' function starts the actual model training. Once the training starts, the training instance
        creates 3 sub-folders in your dataset folder which are:
        - json,  where the JSON configuration file for using your trained model is stored
        - models, where your trained models are stored once they are generated after each improved experiments
        - cache , where temporary traing configuraton files are stored
        :return:
        """

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
                    for batch_i, (data, anns) in tqdm(enumerate(self.__train_loader)):
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
        """
        'setModelTypeAsYOLOv3()' is used to set the model type to the YOLOv3 model.
        :return:
        """
        self.__model_type = "yolov3"

    def setModelTypeAsTinyYOLOv3(self) -> None:
        """
        'setModelTypeAsTinyYOLOv3()' is used to set the model type to the TinyYOLOv3 model.
        :return:
        """
        self.__model_type = "tiny-yolov3"
    
    def setModelPath(self, model_path: str):
        if os.path.isfile(model_path):
            extension_check(model_path)
            self.__model_path = model_path
            self.__model_loaded = False
        else:
            raise ValueError(
                        "invalid path, path not pointing to the weightfile."
                    ) from None
        self.__model_path = model_path
    
    def setJsonPath(self, configuration_json: str):
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


class CustomVideoObjectDetection:
    """
    This is the custom objects detection class for videos and camera live stream inputs in the ImageAI library. It provides support for YOLOv3 and TinyYOLOv3 object detection networks. After instantiating this class, you can set it's properties and
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
        self.__detector = CustomObjectDetection()

    def setModelTypeAsYOLOv3(self):
        self.__detector.setModelTypeAsYOLOv3()
    
    def setModelTypeAsTinyYOLOv3(self):
        self.__detector.setModelTypeAsTinyYOLOv3()

    def setModelPath(self, model_path: str):
        extension_check(model_path)
        self.__detector.setModelPath(model_path)
    
    def setJsonPath(self, configuration_json: str):
        self.__detector.setJsonPath(configuration_json)

    def loadModel(self):
        self.__detector.loadModel()
    
    def useCPU(self):
        self.__detector.useCPU()

    def detectObjectsFromVideo(self, input_file_path="", camera_input=None, output_file_path="", frames_per_second=20,
                               frame_detection_interval=1, minimum_percentage_probability=40, log_progress=False,
                               display_percentage_probability=True, display_object_name=True, display_box=True, save_detected_video=True,
                               per_frame_function=None, per_second_function=None, per_minute_function=None,
                               video_complete_function=None, return_detected_frame=False, detection_timeout = None):

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
                                display_box=display_box)
                            
                        except Exception as e:
                            warnings.warn()
                    
                    if (save_detected_video == True):
                        output_video.write(detected_copy)

                    if detected_copy is not None and output_objects_array is not None:

                        output_frames_dict[counting] = output_objects_array

                        output_objects_count = {}
                        for eachItem in output_objects_array:
                            eachItemName = eachItem["name"]
                            try:
                                output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                            except:
                                output_objects_count[eachItemName] = 1

                        output_frames_count_dict[counting] = output_objects_count

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

            