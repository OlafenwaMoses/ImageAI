import os, re
from typing import Union
import warnings
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torchvision.models import resnet50, densenet121, mobilenet_v2, inception_v3
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import traceback


classification_models = {
    "resnet50": {
        "model": resnet50(pretrained=False)
    },
    "densenet121": {
        "model": densenet121(pretrained=False)
    },
    "inceptionv3": {
        "model": inception_v3(pretrained=False)
    },
    "mobilenetv2": {
        "model": mobilenet_v2(pretrained=False)
    }
}

class ImageClassification:
    """
    This is the image classification class in the ImageAI library. It allows you to classify objects into all the 1000 different classes in the ImageNet dataset [ https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description ].

    The class provides 4 different classification models which are ResNet50, DensesNet121, InceptionV3 and MobileNetV2.

    The following functions are required to be called before a classification can be made

    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsMobileNetV2(), setModelTypeAsResNet(), setModelTypeAsDenseNet, setModelTypeAsInceptionV3]

    * setModelPath(): This is used to specify the abosulute path to the pretrained model files. Download the files in this release -> https://github.com/OlafenwaMoses/ImageAI/releases/tag/untagged-21782fe939a003386296 [TO DO]

    * useCPU (Optional): If you will like to force the image classification to be performed on CPU, call this function.

    * loadModel(): Used to load the pretrained model weights

    * classifyImage(): Used for classifying an image.

    """
    def __init__(self) -> None:
        self.__model_type:str = None
        self.__model: Union[resnet50, densenet121, mobilenet_v2, inception_v3] = None
        self.__model_path: str = None
        self.__classes_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet_classes.txt")
        self.__model_loaded = False
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__classes = []
    
    def setModelPath(self, path: str):
        if os.path.isfile(path):
            self.__model_path = path
        else:
            raise ValueError(
            f"The path '{path}' isn't a valid file. Ensure you specify the path to a valid file."
            )

    def __load_classes(self) -> List[str]:
        with open(self.__classes_path) as f:
            self.__classes = [c.strip() for c in f.readlines()]
    
    def __load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Loads image/images from the given path. If image_path is a directory, this
        function only load the images in the directory (it does not visit the sub-
        directories). This function also convert the loaded image/images to the
        specification expected by the MobileNetV2 architecture.
        """
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

        return torch.stack(images)

    def setModelTypeAsResNet50(self):
        if self.__model_type == None:
            self.__model_type = "resnet50"

    def setModelTypeAsDenseNet121(self):
        if self.__model_type == None:
            self.__model_type = "densenet121"
    
    def setModelTypeAsInceptionV3(self):
        if self.__model_type == None:
            self.__model_type = "inceptionv3"
    
    def setModelTypeAsMobileNetV2(self):
        if self.__model_type == None:
            self.__model_type = "mobilenetv2"
    
    def useCPU(self):
        self.__device = "cpu"
        if self.__model_loaded:
            self.__model_loaded = False
            self.loadModel()

    def loadModel(self):
        if not self.__model_loaded:
            try:
                if self.__model_path == None:
                    raise ValueError(
                        "Model path not specified. Call '.setModelPath()' and parse the path to the model file before loading the model."
                    )
                
                if self.__model_type in classification_models.keys():
                    self.__model = classification_models[self.__model_type]["model"]
                else:
                    raise ValueError(
                        f"Model type '{self.__model_type}' not supported."
                    )
                
                state_dict = torch.load(self.__model_path, map_location=self.__device)
                if self.__model_type == "densenet121":
                    # '.'s are no longer allowed in module names, but previous densenet layers
                    # as provided by the Pytorch's model zoon has names that uses '.'s.
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

                self.__model.load_state_dict(
                        state_dict
                    )
                self.__model_loaded = True
                self.__model.eval()
                self.__load_classes()
            except Exception:
                print(traceback.print_exc())
                print("Weight loading failed.\nEnsure the model path is"
                    " set and the weight file is in the specified model path.")
                
                

    def classifyImage(self, image_input: Union[str, np.ndarray, Image.Image], result_count: int=5) -> Tuple[List[str], List[float]]:
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
                    (self.__classes[topN_catid[i][j]], topN_prob[i][j].item()*100)
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
    

    
    