from imageai.Prediction.Custom import ModelTraining
import os
import pytest
import shutil
import keras

main_folder = os.getcwd()
sample_dataset = os.path.join(main_folder, "data-datasets", "idenprof")
sample_dataset_json_folder = os.path.join(sample_dataset, "json")
sample_dataset_models_folder = os.path.join(sample_dataset, "models")


@pytest.fixture
def clear_keras_session():
    try:
        keras.backend.clear_session()
    except:
        None


@pytest.mark.training
@pytest.mark.training_resnet
@pytest.mark.resnet
@pytest.mark.recognition
def test_resnet_training():

    trainer = ModelTraining()
    trainer.setModelTypeAsResNet()
    trainer.setDataDirectory(data_directory=sample_dataset)
    trainer.trainModel(num_objects=10, num_experiments=1, enhance_data=True, batch_size=16, show_network_summary=True)

    assert os.path.isdir(sample_dataset_json_folder)
    assert os.path.isdir(sample_dataset_models_folder)
    assert os.path.isfile(os.path.join(sample_dataset_json_folder, "model_class.json"))
    assert (len(os.listdir(sample_dataset_models_folder)) > 0)
    shutil.rmtree(os.path.join(sample_dataset_json_folder))
    shutil.rmtree(os.path.join(sample_dataset_models_folder))


@pytest.mark.training
@pytest.mark.training_squeezenet
@pytest.mark.squeezenet
@pytest.mark.recognition
def test_squeezenet_training():

    trainer = ModelTraining()
    trainer.setModelTypeAsSqueezeNet()
    trainer.setDataDirectory(data_directory=sample_dataset)
    trainer.trainModel(num_objects=10, num_experiments=1, enhance_data=True, batch_size=16, show_network_summary=True)

    assert os.path.isdir(sample_dataset_json_folder)
    assert os.path.isdir(sample_dataset_models_folder)
    assert os.path.isfile(os.path.join(sample_dataset_json_folder, "model_class.json"))
    assert (len(os.listdir(sample_dataset_models_folder)) > 0)
    shutil.rmtree(os.path.join(sample_dataset_json_folder))
    shutil.rmtree(os.path.join(sample_dataset_models_folder))


@pytest.mark.training
@pytest.mark.training_inception_v3
@pytest.mark.inception_v3
@pytest.mark.recognition
def test_inception_v3_training():

    trainer = ModelTraining()
    trainer.setModelTypeAsInceptionV3()
    trainer.setDataDirectory(data_directory=sample_dataset)
    trainer.trainModel(num_objects=10, num_experiments=1, enhance_data=True, batch_size=4, show_network_summary=True)

    assert os.path.isdir(sample_dataset_json_folder)
    assert os.path.isdir(sample_dataset_models_folder)
    assert os.path.isfile(os.path.join(sample_dataset_json_folder, "model_class.json"))
    assert (len(os.listdir(sample_dataset_models_folder)) > 0)
    shutil.rmtree(os.path.join(sample_dataset_json_folder))
    shutil.rmtree(os.path.join(sample_dataset_models_folder))
    





