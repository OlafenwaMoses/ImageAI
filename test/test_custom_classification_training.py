import os, sys
import cv2
import shutil
from PIL import Image
import pytest
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(dirname(os.path.abspath(__file__)))))
from imageai.Classification.Custom import ClassificationModelTrainer, CustomImageClassification

test_folder = dirname(os.path.abspath(__file__))


classification_dataset = os.path.join(
    test_folder,
    "data-datasets",
    "idenprof"
)

pretrained_models_folder = os.path.join(
    test_folder,
    "data-models"
)


@pytest.mark.parametrize(
    "transfer_learning",
    [
        (os.path.join(
            pretrained_models_folder,
            "resnet50-19c8e357.pth"
        )),
        (None),
    ]
)
def test_resnet50_training(transfer_learning):

    models_dir = os.path.join(
        classification_dataset,
        "models"
    )
    if os.path.isdir(
        models_dir
    ):
        shutil.rmtree(models_dir)

    trainer = ClassificationModelTrainer()
    trainer.setModelTypeAsResNet50()
    trainer.setDataDirectory(data_directory=classification_dataset)
    trainer.trainModel(
        num_experiments=1,
        batch_size=2,
        transfer_from_model=transfer_learning)

    assert os.path.isdir(models_dir) == True
    assert os.path.isfile(
        os.path.join(
            models_dir, "idenprof_model_classes.json"
        )
    ) == True
    
    model_found = False
    for file in os.listdir(models_dir):
        if file.endswith(".pt"):
            model_found = True
    assert model_found == True


@pytest.mark.parametrize(
    "transfer_learning",
    [
        (os.path.join(
            pretrained_models_folder,
            "densenet121-a639ec97.pth"
        )),
        (None),
    ]
)
def test_densenet121_training(transfer_learning):

    models_dir = os.path.join(
        classification_dataset,
        "models"
    )
    if os.path.isdir(
        models_dir
    ):
        shutil.rmtree(models_dir)

    trainer = ClassificationModelTrainer()
    trainer.setModelTypeAsDenseNet121()
    trainer.setDataDirectory(data_directory=classification_dataset)
    trainer.trainModel(
        num_experiments=1,
        batch_size=2,
        transfer_from_model=transfer_learning)

    assert os.path.isdir(models_dir) == True
    assert os.path.isfile(
        os.path.join(
            models_dir, "idenprof_model_classes.json"
        )
    ) == True
    model_found = False
    for file in os.listdir(models_dir):
        if file.endswith(".pt"):
            model_found = True
    assert model_found == True



@pytest.mark.parametrize(
    "transfer_learning",
    [
        (os.path.join(
            pretrained_models_folder,
            "inception_v3_google-1a9a5a14.pth"
        )),
        (None),
    ]
)
def test_inceptionv3_training(transfer_learning):

    models_dir = os.path.join(
        classification_dataset,
        "models"
    )
    if os.path.isdir(
        models_dir
    ):
        shutil.rmtree(models_dir)

    trainer = ClassificationModelTrainer()
    trainer.setModelTypeAsInceptionV3()
    trainer.setDataDirectory(data_directory=classification_dataset)
    trainer.trainModel(
        num_experiments=1,
        batch_size=2,
        transfer_from_model=transfer_learning)

    assert os.path.isdir(models_dir) == True
    assert os.path.isfile(
        os.path.join(
            models_dir, "idenprof_model_classes.json"
        )
    ) == True
    model_found = False
    for file in os.listdir(models_dir):
        if file.endswith(".pt"):
            model_found = True
    assert model_found == True


@pytest.mark.parametrize(
    "transfer_learning",
    [
        (os.path.join(
            pretrained_models_folder,
            "mobilenet_v2-b0353104.pth"
        )),
        (None),
    ]
)
def test_mobilenetv2_training(transfer_learning):

    models_dir = os.path.join(
        classification_dataset,
        "models"
    )
    if os.path.isdir(
        models_dir
    ):
        shutil.rmtree(models_dir)

    trainer = ClassificationModelTrainer()
    trainer.setModelTypeAsMobileNetV2()
    trainer.setDataDirectory(data_directory=classification_dataset)
    trainer.trainModel(
        num_experiments=1,
        batch_size=2,
        transfer_from_model=transfer_learning)

    assert os.path.isdir(models_dir) == True
    assert os.path.isfile(
        os.path.join(
            models_dir, "idenprof_model_classes.json"
        )
    ) == True
    model_found = False
    for file in os.listdir(models_dir):
        if file.endswith(".pt"):
            model_found = True
    assert model_found == True
