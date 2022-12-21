import os, sys
import shutil
import pytest
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(dirname(os.path.abspath(__file__)))))
from imageai.Detection.Custom import DetectionModelTrainer

test_folder = dirname(os.path.abspath(__file__))


detection_dataset = os.path.join(
    test_folder,
    "data-datasets",
    "number-plate"
)

pretrained_models_folder = os.path.join(
    test_folder,
    "data-models"
)

def delete_cache(dirs: list):
    for dir in dirs:
        if os.path.isdir(dir):
            shutil.rmtree(dir)

@pytest.mark.parametrize(
    "transfer_learning",
    [
        (os.path.join(
            pretrained_models_folder,
            "yolov3.pt"
        )),
        (None),
    ]
)
def test_yolov3_training(transfer_learning):
    json_dir = os.path.join(detection_dataset, "json")
    json_file = os.path.join(json_dir, "number-plate_yolov3_detection_config.json")
    models_dir = os.path.join(detection_dataset, "models")

    delete_cache([json_dir, models_dir])

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=detection_dataset)
    trainer.setTrainConfig(object_names_array=["number-plate"], batch_size=2, num_experiments=2, train_from_pretrained_model=transfer_learning)
    trainer.trainModel()

    
    assert os.path.isfile(json_file)
    assert len([file for file in os.listdir(models_dir) if file.endswith(".pt")]) > 0

    delete_cache([json_dir, models_dir])

@pytest.mark.parametrize(
    "transfer_learning",
    [
        (os.path.join(
            pretrained_models_folder,
            "tiny-yolov3.pt"
        )),
        (None),
    ]
)
def test_tiny_yolov3_training(transfer_learning):
    json_dir = os.path.join(detection_dataset, "json")
    json_file = os.path.join(json_dir, "number-plate_tiny-yolov3_detection_config.json")
    models_dir = os.path.join(detection_dataset, "models")

    delete_cache([json_dir, models_dir])

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsTinyYOLOv3()
    trainer.setDataDirectory(data_directory=detection_dataset)
    trainer.setTrainConfig(object_names_array=["number-plate"], batch_size=2, num_experiments=2, train_from_pretrained_model=transfer_learning)
    trainer.trainModel()

    
    assert os.path.isfile(json_file)
    assert len([file for file in os.listdir(models_dir) if file.endswith(".pt")]) > 0

    delete_cache([json_dir, models_dir])