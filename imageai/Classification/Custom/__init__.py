import time, warnings
import os
import copy
import re
import json
from typing import List, Tuple, Union
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms
from torchvision.models import mobilenet_v2, inception_v3, resnet50, densenet121
from torchvision.models.inception import InceptionOutputs

from .data_transformation import data_transforms1, data_transforms2
from .training_params import resnet50_train_params, densenet121_train_params, inception_v3_train_params, mobilenet_v2_train_params
from tqdm import tqdm

from ...backend_check.model_extension import extension_check



class ClassificationModelTrainer():
    """
        This is the Classification Model training class, that allows you to define a deep learning network
        from the 4 available networks types supported by ImageAI which are MobileNetv2, ResNet50,
        InceptionV3 and DenseNet121 and then train on custom image data.
    """

    def __init__(self) -> None:
        self.__model_type = ""
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__data_dir = ""
        self.__data_loaders = None
        self.__class_names = None
        self.__dataset_sizes = None
        self.__dataset_name = ""
        self.__model = None
        self.__optimizer = None
        self.__lr_scheduler = None
        self.__loss_fn = nn.CrossEntropyLoss()
        self.__transfer_learning_mode = "fine_tune_all"
        self.__model_path = ""
        self.__training_params = None

    def __set_training_param(self) -> None:
        if not self.__model_type:
            raise RuntimeError("The model type is not set!!!")
        self.__model = self.__training_params["model"]
        optimizer = self.__training_params["optimizer"]
        lr_decay_rate = self.__training_params["lr_decay_rate"]
        lr_step_size = self.__training_params["lr_step_size"]
        lr = self.__training_params["lr"]
        weight_decay = self.__training_params["weight_decay"]

        if self.__model_path:
            self.__set_transfer_learning_mode()
            print("==> Transfer learning enabled")
        
        # change the last linear layer to have output features of
        # same size as the number of unique classes in the new
        # dataset.
        if self.__model_type == "mobilenet_v2":
            in_features = self.__model.classifier[1].in_features
            self.__model.classifier[1] = nn.Linear(in_features, len(self.__class_names))
        elif self.__model_type == "densenet121":
            in_features = self.__model.classifier.in_features
            self.__model.classifier = nn.Linear(in_features, len(self.__class_names))
        else:
            in_features = self.__model.fc.in_features
            self.__model.fc = nn.Linear(in_features, len(self.__class_names))

        self.__model.to(self.__device)
        self.__optimizer = optimizer(
                    self.__model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay
                )
        if lr_decay_rate and lr_step_size:
            self.__lr_scheduler = lr_scheduler.StepLR(
                                self.__optimizer,
                                gamma=lr_decay_rate,
                                step_size=lr_step_size
                            )

    def __set_transfer_learning_mode(self) -> None:

        state_dict = torch.load(self.__model_path)
        if self.__model_type == "densenet121":
            # '.'s are no longer allowed in module names, but previous densenet layers
            # as provided by the pytorch organization has names that uses '.'s.
            pattern = re.compile(
                    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\."
                    "(?:weight|bias|running_mean|running_var))$"
                    )
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        self.__model.load_state_dict(state_dict)
        self.__model.to(self.__device)

        if self.__transfer_learning_mode == "freeze_all":
            for param in self.__model.parameters():
                param.requires_grad = False

    def __load_data(self, batch_size : int = 8) -> None:
        
        if not self.__data_dir:
            raise RuntimeError("The dataset directory not yet set.")
        image_dataset = {
                        x:datasets.ImageFolder(
                                os.path.join(self.__data_dir, x),
                                data_transforms2[x] if self.__model_type=="inception_v3" else data_transforms1[x]
                            )
                        for x in ["train", "test"]
                    }
        self.__data_loaders = {
                        x:torch.utils.data.DataLoader(
                                image_dataset[x], batch_size=batch_size,
                                shuffle=True
                            )
                        for x in ["train", "test"]
                    }
        self.__dataset_sizes = {x:len(image_dataset[x]) for x in ["train", "test"]}
        self.__class_names = image_dataset["train"].classes
        self.__dataset_name = os.path.basename(self.__data_dir.rstrip(os.path.sep))

    def setDataDirectory(self, data_directory : str = "") -> None:
        """
        Sets the directory that contains the training and test dataset. The data directory should contain 'train' and 'test' subdirectories
        for the training and test datasets.

        In each of these subdirectories, each object must have a dedicated folder and the folder containing images for the object.

        The structure of the 'test' and 'train' folder must be as follows:
        
        >> train >> class1 >> class1_train_images
                    >> class2 >> class2_train_images
                    >> class3 >> class3_train_images
                    >> class4 >> class4_train_images
                    >> class5 >> class5_train_images
        >> test >> class1 >> class1_test_images
                >> class2 >> class2_test_images
                >> class3 >> class3_test_images
                >> class4 >> class4_test_images
                >> class5 >> class5_test_images

        """
        if os.path.isdir(data_directory):
            self.__data_dir = data_directory
            return
        raise ValueError("expected a path to a directory")

    def setModelTypeAsMobileNetV2(self) -> None:
        """
        'setModelTypeAsMobileNetV2()' is used to set the model type to the MobileNetV2 model.
        :return:
        """
        self.__model_type = "mobilenet_v2"
        self.__training_params = mobilenet_v2_train_params()

    def setModelTypeAsResNet50(self) -> None:
        """
        'setModelTypeAsResNet50()' is used to set the model type to the ResNet50 model.
        :return:
        """
        self.__model_type = "resnet50"
        self.__training_params = resnet50_train_params()

    def setModelTypeAsInceptionV3(self) -> None:
        """
        'setModelTypeAsInceptionV3()' is used to set the model type to the InceptionV3 model.
        :return:
        """
        self.__model_type = "inception_v3"
        self.__training_params = inception_v3_train_params()

    def setModelTypeAsDenseNet121(self) -> None:
        """
        'setModelTypeAsDenseNet()' is used to set the model type to the DenseNet model.
        :return:
        """
        self.__model_type = "densenet121"
        self.__training_params = densenet121_train_params()

    def freezeAllLayers(self) -> None:
        """
        Set the transfer learning mode to freeze all layers.

        NOTE: The last layer (fully connected layer) is trainable.
        """
        self.__transfer_learning_mode = "freeze_all"

    def fineTuneAllLayers(self) -> None:
        """
        Sets the transfer learning mode to fine-tune the pretrained weights
        """
        self.__transfer_learning_mode = "fine_tune_all"

    def trainModel(
                self,
                num_experiments : int = 100,
                batch_size : int = 8,
                model_directory  : str = None,
                transfer_from_model: str = None,
                verbose : bool = True
            ) -> None:
        
        """
        'trainModel()' function starts the model actual training. It accepts the following values:
        - num_experiments: Also known as epochs, is the number of times the network will process all the images in the training dataset
        - batch_size: The number of image data that will be loaded into memory at once during training
        - model_directory: Location where json mapping and trained models will be saved
        - transfer_from_model: Path to a pre-trained imagenet model that corresponds to the training model type
        - verbose: Option to enable/disable training logs
        
        :param num_experiments:
        :param batch_size:
        :model_directory:
        :transfer_from_model:
        :verbose:
        :return:
        """

        # Load dataset
        self.__load_data(batch_size)

        # Check and effect transfer learning if enabled
        if transfer_from_model:
            extension_check(transfer_from_model)
            self.__model_path = transfer_from_model

        # Load training parameters for the specified model type
        self.__set_training_param()

        
        # Create output directory to save trained models and json mappings
        if not model_directory:
            model_directory = os.path.join(self.__data_dir, "models")

        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
        
        # Dump class mappings to json file
        with open(os.path.join(model_directory, f"{self.__dataset_name}_model_classes.json"), "w") as f:
            classes_dict = {}
            class_list = sorted(self.__class_names)
            for i in range(len(class_list)):
                classes_dict[str(i)] = class_list[i]
            json.dump(classes_dict, f)

        # Prep model weights for training
        since = time.time()

        best_model_weights = copy.deepcopy(self.__model.state_dict())
        best_acc = 0.0
        prev_save_name, recent_save_name = "", ""

        # Device check and log
        print("=" * 50)
        print("Training with GPU") if self.__device == "cuda" else print("Training with CPU. This might cause slower train.")
        print("=" * 50)


        for epoch in range(num_experiments):
            if verbose:
                print(f"Epoch {epoch + 1}/{num_experiments}", "-"*10, sep="\n")

            # each epoch has a training and test phase
            for phase in ["train", "test"]:
                if phase == "train":
                    self.__model.train()
                else:
                    self.__model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate on the dataset in batches
                for imgs, labels in tqdm(self.__data_loaders[phase]):
                    imgs = imgs.to(self.__device)
                    labels = labels.to(self.__device)

                    self.__optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        output = self.__model(imgs)
                        if self.__model_type == "inception_v3" and type(output) == InceptionOutputs:
                            output = output[0]
                        _, preds = torch.max(output, 1)
                        loss = self.__loss_fn(output, labels)

                        if phase=="train":
                            loss.backward()
                            self.__optimizer.step()
                    running_loss += loss.item() * imgs.size(0)
                    running_corrects += torch.sum(preds==labels.data)

                # Compute accuracy and loss metrics post epoch training
                if phase == "train" and isinstance(self.__lr_scheduler, torch.optim.lr_scheduler.StepLR):
                    self.__lr_scheduler.step()

                epoch_loss = running_loss / self.__dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.__dataset_sizes[phase]

                if verbose:
                    print(f"{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    recent_save_name = self.__model_type+f"-{self.__dataset_name}-test_acc_{best_acc:.5f}_epoch-{epoch}.pt"
                    if prev_save_name:
                        os.remove(os.path.join(model_directory, prev_save_name))
                    best_model_weights = copy.deepcopy(self.__model.state_dict())
                    torch.save(
                            best_model_weights, os.path.join(model_directory, recent_save_name)
                        )
                    prev_save_name = recent_save_name
            

        time_elapsed = time.time() - since
        print(f"Training completed in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best test accuracy: {best_acc:.4f}")


class CustomImageClassification:
    """
    An implementation that allows for easy classification of images
    using the state of the art computer vision classification model
    trained on custom data.

    The class provides 4 different classification models which are ResNet50, DensesNet121, InceptionV3 and MobileNetV2.

    The following functions are required to be called before a classification can be made

    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsMobileNetV2(), setModelTypeAsResNet(), setModelTypeAsDenseNet, setModelTypeAsInceptionV3]

    * setModelPath: This is used to specify the absolute path to the trained model file.

    * setJsonPath: This is used to specify the absolute path to the
    json file saved during the training of the custom model.

    * useCPU (Optional): If you will like to force the image classification to be performed on CPU, call this function.

    * loadModel: Used to load the trained model weights and json data.

    * classifyImage(): Used for classifying an image.
    """
    def __init__(self) -> None:
        self.__model = None
        self.__model_type = ""
        self.__model_loaded = False
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__json_path = None
        self.__class_names = None
        self.__model_loaded = False

    def __load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        images = []
        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if type(image_input) == str:
            if os.path.isfile(image_input):
                img = Image.open(image_input).convert("RGB")
                images.append(preprocess(img))
            else:
                raise ValueError(f"image path '{image_input}' is not found or a valid file")
        elif type(image_input) == np.ndarray:
            img = Image.fromarray(image_input).convert("RGB")
            images.append(preprocess(img))
        elif "PIL" in str(type(image_input)):
            img = image_input.convert("RGB")
            images.append(preprocess(img))
        else:
            raise ValueError(f"Invalid image input format")

        return torch.stack(images)
    
    def __load_classes(self):
        if self.__json_path:
            with open(self.__json_path, 'r') as f:
                self.__class_names = list(json.load(f).values())
        else:
            raise ValueError("Invalid json path. Set a valid json mapping path by calling the 'setJsonPath()' function")

    def setModelPath(self, path : str) -> None:
        """
        Sets the path to the pretrained weight.
        """
        if os.path.isfile(path):
            extension_check(path)
            self.__model_path = path
            self.__model_loaded = False
        else:
            raise ValueError(
                f"The path '{path}' isn't a valid file. Ensure you specify the path to a valid trained model file."
            )
    
    def setJsonPath(self, path : str) -> None:
        """
        Sets the path to the pretrained weight.
        """
        if os.path.isfile(path):
            self.__json_path = path
        else:
            raise ValueError(
            "parameter path should be a valid path to the json mapping file."
            )

    def setModelTypeAsMobileNetV2(self) -> None:
        """
        'setModelTypeAsMobileNetV2()' is used to set the model type to the MobileNetV2 model.
        :return:
        """
        self.__model_type = "mobilenet_v2"

    def setModelTypeAsResNet50(self) -> None:
        """
        'setModelTypeAsResNet50()' is used to set the model type to the ResNet50 model.
        :return:
        """
        self.__model_type = "resnet50"

    def setModelTypeAsInceptionV3(self) -> None:
        """
        'setModelTypeAsInceptionV3()' is used to set the model type to the InceptionV3 model.
        :return:
        """
        self.__model_type = "inception_v3"

    def setModelTypeAsDenseNet121(self) -> None:
        """
        'setModelTypeAsDenseNet121()' is used to set the model type to the DenseNet121 model.
        :return:
        """
        self.__model_type = "densenet121"
    
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
            self.__load_classes()
            try:
                # change the last layer of the networks to conform to the number
                # of unique classes in the custom dataset used to train the custom
                # model

                if self.__model_type == "resnet50":
                    self.__model = resnet50(pretrained=False)
                    in_features = self.__model.fc.in_features
                    self.__model.fc = nn.Linear(in_features, len(self.__class_names))
                elif self.__model_type == "mobilenet_v2":
                    self.__model = mobilenet_v2(pretrained=False)
                    in_features = self.__model.classifier[1].in_features
                    self.__model.classifier[1] = nn.Linear(in_features, len(self.__class_names))
                elif self.__model_type == "inception_v3":
                    self.__model = inception_v3(pretrained=False)
                    in_features = self.__model.fc.in_features
                    self.__model.fc = nn.Linear(in_features, len(self.__class_names))
                elif self.__model_type == "densenet121":
                    self.__model = densenet121(pretrained=False)
                    in_features = self.__model.classifier.in_features
                    self.__model.classifier = nn.Linear(in_features, len(self.__class_names))
                else:
                    raise RuntimeError("Unknown model type.\nEnsure the model type is properly set.")

                state_dict = torch.load(self.__model_path, map_location=self.__device)

                if self.__model_type == "densenet121":
                    # '.'s are no longer allowed in module names, but previous densenet layers
                    # as provided by the pytorch organization has names that uses '.'s.
                    pattern = re.compile(
                            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\."
                                    "(?:weight|bias|running_mean|running_var))$"
                            )
                    for key in list(state_dict.keys()):
                        res = pattern.match(key)
                        if res:
                            new_key = res.group(1) + res.group(2)
                            state_dict[new_key] = state_dict[key]
                            del state_dict[key]

                self.__model.load_state_dict(state_dict)
                self.__model.to(self.__device).eval()
                self.__model_loaded = True

            except Exception as e:
                raise Exception("Weight loading failed.\nEnsure the model path is"
                    " set and the weight file is in the specified model path.")

    def classifyImage(self, image_input: Union[str, np.ndarray, Image.Image], result_count: int) -> Tuple[List[str], List[float]]:
        """
        'classifyImage()' function is used to classify a given image by receiving the following arguments:
            * image_input: file path, numpy array or PIL image of the input image.
            * result_count (optional) , the number of classifications to be sent which must be whole numbers between 1 and total number of classes the model is trained to classify.

        This function returns 2 arrays namely 'classification_results' and 'classification_probabilities'. The 'classification_results'
        contains possible objects classes arranged in descending of their percentage probabilities. The 'classification_probabilities'
        contains the percentage probability of each object class. The position of each object class in the 'classification_results'
        array corresponds with the positions of the percentage probability in the 'classification_probabilities' array.
        
        :param image_input:
        :param result_count:
        :return classification_results, classification_probabilities:
        """
        if not self.__model_loaded:
            raise RuntimeError(
                "Model not yet loaded. You need to call '.loadModel()' before performing image classification"
            )

        images = self.__load_image(image_input)
        images = images.to(self.__device)
    
        with torch.no_grad():
            output = self.__model(images)
        probabilities = torch.softmax(output, dim=1)
        topN_prob, topN_catid = torch.topk(probabilities, result_count)
        
        predictions = [
                [
                    (self.__class_names[topN_catid[i][j]], topN_prob[i][j].item()*100)
                    for j in range(topN_prob.shape[1])
                ]
                for i in range(topN_prob.shape[0])
            ]
        
        labels_pred = []
        probabilities_pred = []

        for idx, pred in enumerate(predictions):
            for label, score in pred:
                labels_pred.append(label)
                probabilities_pred.append(round(score, 4))
        
        return labels_pred, probabilities_pred