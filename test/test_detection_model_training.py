from imageai.Detection.Custom import DetectionModelTrainer
import os
import shutil
import keras
import pytest

"""

main_folder = os.getcwd()
sample_dataset = os.path.join(main_folder, "data-datasets", "hololens")
sample_dataset_json_folder = os.path.join(sample_dataset, "json")
sample_dataset_models_folder = os.path.join(sample_dataset, "models")
sample_dataset_cache_folder = os.path.join(sample_dataset, "cache")
pretrained_model = os.path.join(main_folder, "data-models", "pretrained-yolov3.h5")


@pytest.fixture
def clear_keras_session():
    try:
        keras.backend.clear_session()
    except:
        None


@pytest.mark.detection
@pytest.mark.training
@pytest.mark.detection_training
@pytest.mark.detection_transfer_learning
@pytest.mark.yolov3
def test_detection_training(clear_keras_session):


    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=sample_dataset)
    trainer.setTrainConfig(object_names_array=["hololens"], batch_size=2, num_experiments=1, train_from_pretrained_model=pretrained_model)
    trainer.trainModel()

    assert os.path.isdir(sample_dataset_json_folder)
    assert os.path.isdir(sample_dataset_models_folder)
    assert os.path.isdir(sample_dataset_cache_folder)
    assert os.path.isfile(os.path.join(sample_dataset_json_folder, "detection_config.json"))
    shutil.rmtree(os.path.join(sample_dataset_json_folder))
    shutil.rmtree(os.path.join(sample_dataset_models_folder))
    shutil.rmtree(os.path.join(sample_dataset_cache_folder))



"""