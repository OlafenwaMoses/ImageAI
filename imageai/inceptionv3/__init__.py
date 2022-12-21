import os, warnings
from pathlib import Path
from typing import List, Tuple

import torch, torchvision
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

warnings.filterwarnings("once", category=ResourceWarning)

class InceptionV3Pretrained:
    """
    An implementation that allows for easy classification of images
    using the state of the art MobileNet computer vision model.
    """
    def __init__(self, label_path : str) -> None:
        self.__model = torchvision.models.inception_v3(pretrained=False)
        self.__classes = self.__load_classes(label_path)
        self.__has_loaded_weights = False
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model_path = ""
        
    def __load_classes(self, path : str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def __load_image(self, image_path : str) -> Tuple[List[str], torch.Tensor]:
        """
        Loads image/images from the given path. If image_path is a directory, this
        function only load the images in the directory (it does not visit the sub-
        directories). This function also convert the loaded image/images to the
        specification expected by the MobileNetV2 architecture.
        """
        allowed_file_extensions = ["jpg", "jpeg", "png"]
        images = []
        fnames = []
        preprocess = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if os.path.isfile(image_path):
            img = Image.open(image_path).convert("RGB")
            images.append(preprocess(img))
            fnames.append(os.path.basename(image_path))

        elif os.path.isdir(image_path):
            for file in os.listdir(image_path):
                if os.path.isfile(os.path.join(image_path, file)) and\
                        file.rsplit('.')[-1].lower() in allowed_file_extensions:
                            img = Image.open(os.path.join(image_path, file)).convert("RGB")
                            images.append(preprocess(img))
                            fnames.append(file)
        if images:
            return fnames, torch.stack(images)
        raise RuntimeError(
                f"Error loading images from {os.path.abspath(image_path)}."
                "\nEnsure the folder contains images,"
                " allowed file extensions are .jpg, .jpeg, .png"
            )

    # properties
    model_path = property(
                fget=lambda self : self.__model_path,
                fset=lambda self, path: self.set_model_path(path),
                doc="Path containing the pretrained weight."
            )

    def set_model_path(self, path : str) -> None:
        """
        Sets the path to the pretrained weight.
        """
        if os.path.isfile(path):
            self.__model_path = path
            self.__has_loaded_weights = False
        else:
            raise ValueError(
            "parameter path should be a path to the pretrianed weight file."
            )

    def load_model(self) -> None:
        """
        Loads the mobilenet vison weight into the model architecture.
        """
        if not self.__has_loaded_weights:
            try:
                self.__model.load_state_dict(
                        torch.load(self.__model_path, map_location=self.__device)
                    )
                self.__has_loaded_weights = True
                self.__model.eval()
            except Exception:
                print("Weight loading failed.\nEnsure the model path is"
                    " set and the weight file is in the specified model path.")

    def classify(self, image_path : str, top_n : int = 5, verbose : bool = True) -> List[List[Tuple[str, str]]]:
        """
        Classfies image/images according to the classes provided by imagenet.

        Parameters:
        -----------
            image_path: a path to a single image or a path to a directory containing
                        images. If image_path is a path to a file, this functions
                        classifies the image according to the categories provided
                        by imagenet, else, if image_path is a path to a directory
                        that contains images, this function classifies all images in
                        the given directory (it doesn't visit the subdirectories).

            top_n: number of top predictions to return.
            verbose: if true, it prints the top_n predictions.
        """
        if not self.__has_loaded_weights:
            if self.__model_path:
                warnings.warn(
                        "Model path has changed but pretrained weights in the"
                        " new path are yet to be loaded.",
                        ResourceWarning
                    )
            else:
                warnings.warn(
                        "Model path isn't set, pretrained weights aren't used.",
                        ResourceWarning
                    )

        fnames, images = self.__load_image(image_path)
        images = images.to(self.__device)
        print(images.shape)
    
        with torch.no_grad():
            output = self.__model(images)
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        with open(os.path.join(str(Path(__file__).resolve().parent.parent), "imagenet_classes.txt")) as f:
            categories = [c.strip() for c in f.readlines()]
        predictions = [
                [
                    (categories[top5_catid[i][j]], f"{top5_prob[i][j].item()*100:.5f}%")
                    for j in range(top5_prob.shape[1])
                ]
                for i in range(top5_prob.shape[0])
            ]

        if verbose:
            for idx, pred in enumerate(predictions):
                print("-"*50, f"Top 5 predictions for {fnames[idx]}", "-"*50, sep="\n")
                for label, score in pred:
                    print(f"\t{label}:{score: >10}")
                print("-"*50, "\n")
        return predictions

