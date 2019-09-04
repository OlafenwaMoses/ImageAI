from imageai.Prediction.Custom import CustomImagePrediction
import os
import pytest
from os.path import dirname
import keras


main_folder = os.getcwd()

all_images = os.listdir(os.path.join(main_folder, "data-images"))
all_images_array = []


def images_to_image_array():
    for image in all_images:
        all_images_array.append(os.path.join(main_folder, "data-images", image))



@pytest.mark.recognition
@pytest.mark.resnet
@pytest.mark.recognition_custom
def test_custom_recognition_model_resnet():


    predictor = CustomImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_resnet.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.recognition
@pytest.mark.resnet
@pytest.mark.recognition_custom
def test_custom_recognition_full_model_resnet():

    predictor = CustomImagePrediction()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_full_resnet_ex-001_acc-0.119792.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadFullModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.recognition
@pytest.mark.densenet
@pytest.mark.recognition_custom
def test_custom_recognition_model_densenet():

    predictor = CustomImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_densenet-0.763500.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

