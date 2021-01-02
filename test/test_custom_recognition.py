from imageai.Prediction.Custom import CustomImageClassification
import pytest
from os.path import dirname
import os


main_folder = os.getcwd()

all_images = os.listdir(os.path.join(main_folder, "data-images"))
all_images_array = []


def images_to_image_array():
    for image in all_images:
        all_images_array.append(os.path.join(main_folder, "data-images", image))




def test_custom_recognition_model_resnet():


    predictor = CustomImageClassification()
    predictor.setModelTypeAsResNet50()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_resnet.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)


def test_custom_recognition_model_densenet():

    predictor = CustomImageClassification()
    predictor.setModelTypeAsDenseNet121()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_densenet-0.763500.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

