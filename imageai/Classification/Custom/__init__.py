import time, warnings
import os
import copy
import re
import json

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets

from .data_transformation import data_transforms1, data_transforms2
from .training_params import resnet50_train_params
from tqdm import tqdm



class ClassificationModelTrainer():

    def __init__(self, use_transfer_learning : bool = False) -> None:
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
        self.__transfer_learning = use_transfer_learning
        self.__transfer_learning_mode = ""
        self.__model_path = ""
        self.__training_params = None

    def __set_training_param(self) -> None:
        """
        Sets the required training parameters for the specified vision model.
        The default parameters used are the ones specified by the authors in
        their research paper.
        """
        if not self.__model_type:
            raise RuntimeError("The model type is not set!!!")
        self.__model = self.__training_params["model"]
        optimizer = self.__training_params["optimizer"]
        lr_decay_rate = self.__training_params["lr_decay_rate"]
        lr_step_size = self.__training_params["lr_step_size"]
        lr = self.__training_params["lr"]
        weight_decay = self.__training_params["weight_decay"]

        if self.__transfer_learning and self.__model_path:
            self.__set_transfer_learning_mode()
        
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
        """
        Put all layers of the neural network in right state for transfer
        learning.
        """
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

        if self.__transfer_learning_mode == "freeze_all":
            for param in self.__model.parameters():
                param.requires_grad = False

    def __load_data(self, batch_size : int = 8) -> None:
        """
        Load the data in the specified data directory.
        The data directory should contain 'train' and 'val' subdirectories
        for the training and validation datasets. The subdirectories in the
        'train' and 'val' should be named  after each unique classes.
        """
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

    # properties
    model_type = property(
                fget=lambda self: self.__model_type,
                doc="the current computer vision model being used."
            )
    data_directory = property(
                fget=lambda self : self.__data_dir,
                fset=lambda self, path : self.set_data_directory(path),
                doc="Path to the directory containing the training and validation data."
            )
    model_path = property(
                fget=lambda self : self.__model_path,
                fset=lambda self, path : self.set_model_path(path),
                doc="Path to the binary file containing pretrained weights."
            )

    def setDataDirectory(self, data_dir : str = "") -> None:
        """
        Sets the directory that contains the training and validation 
        dataset.
        """
        if os.path.isdir(data_dir):
            self.__data_dir = data_dir
            return
        raise ValueError("expected a path to a directory")

    def set_model_path(self, path : str = "") -> None:
        """
        Sets the path to the binary file containing the pretrained
        weights to be used for transfer learning.
        """
        if os.path.isfile(path):
            self.__model_path = path
            return
        raise ValueError(
                "Ensure the path points to the binary file containing the"
                " pretrained weights."
            )

    def setModelAsMobileNetV2(self) -> None:
        self.__model_type = "mobilenet_v2"

    def setModelAsResNet50(self) -> None:
        self.__model_type = "resnet50"
        self.__training_params = resnet50_train_params()

    def setModelAsInceptionV3(self) -> None:
        self.__model_type = "inception_v3"

    def setModelAsDenseNet121(self) -> None:
        self.__model_type = "densenet121"

    def freeze_all_layers(self) -> None:
        """
        Set the transfer learning mode to freeze all layers.

        NOTE: The last layer (fully connected layer) is trainable.
        """
        self.__transfer_learning_mode = "freeze_all"

    def fine_tune_all_layers(self) -> None:
        """
        Sets the transfer learning mode to fine-tune the pretrained weights
        """
        self.__transfer_learning_mode = "fine_tune_all"

    def trainModel(
                self,
                num_experiments : int = 100,
                batch_size : int = 8,
                model_directory  : str = "",
                verbose : bool = True
            ) -> None:

        if self.__transfer_learning and not self.__model_path:
            warnings.warn(
                    "Path to pretrained weights to use for transfer learnin isn't"
                    " set.\nDefaulting to training all layers from scratch...",
                    ResourceWarning
                )

        self.__load_data(batch_size)
        self.__set_training_param()

        if model_directory:
            model_directory = os.path.join(model_directory, "models")
        else:
            model_directory = os.path.join(self.__data_dir, "models")

        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
        
        with open(os.path.join(model_directory, f"{self.__dataset_name}_model_classes.json"), "w") as f:
            classes_dict = {}
            class_list = sorted(self.__class_names)
            for i in range(len(class_list)):
                classes_dict[str(i)] = class_list[i]
            json.dump(classes_dict, f)

        since = time.time()

        best_model_weights = copy.deepcopy(self.__model.state_dict())
        best_acc = 0.0
        prev_save_name, recent_save_name = "", ""

        for epoch in range(num_experiments):
            if verbose:
                print(f"Epoch {epoch}/{num_experiments - 1}", "-"*10, sep="\n")

            # each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    self.__model.train()
                else:
                    self.__model.eval()

                running_loss = 0.0
                running_corrects = 0

                for imgs, labels in tqdm(self.__data_loaders[phase]):
                    imgs = imgs.to(self.__device)
                    labels = labels.to(self.__device)

                    self.__optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        output = self.__model(imgs)
                        if self.__model_type == "inception_v3":
                            output = output.logits
                        _, preds = torch.max(output, 1)
                        loss = self.__loss_fn(output, labels)

                        if phase=="train":
                            loss.backward()
                            self.__optimizer.step()
                    running_loss += loss.item() * imgs.size(0)
                    running_corrects += torch.sum(preds==labels.data)

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
            if verbose:
                print()

        time_elapsed = time.time() - since
        print(f"Training completed in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best test accuracy: {best_acc:.4f}")
