import os, sys
import cv2
from PIL import Image
import pytest
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(dirname(os.path.abspath(__file__)))))
from imageai.Classification import ImageClassification

test_folder = dirname(os.path.abspath(__file__))



@pytest.mark.parametrize(
    "image_input",
    [
        (os.path.join(test_folder, test_folder, "data-images", "1.jpg")),
        (cv2.imread(os.path.join(test_folder, test_folder, "data-images", "1.jpg"))),
        (Image.open(os.path.join(test_folder, test_folder, "data-images", "1.jpg"))),
    ]
)
def test_recognition_model_mobilenetv2(image_input):

    predictor = ImageClassification()
    predictor.setModelTypeAsMobileNetV2()
    predictor.setModelPath(os.path.join(test_folder, "data-models", "mobilenet_v2-b0353104.pth"))
    predictor.loadModel()
    predictions, probabilities = predictor.classifyImage(image_input=image_input)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)


@pytest.mark.parametrize(
    "image_input",
    [
        (os.path.join(test_folder, test_folder, "data-images", "1.jpg")),
        (cv2.imread(os.path.join(test_folder, test_folder, "data-images", "1.jpg"))),
        (Image.open(os.path.join(test_folder, test_folder, "data-images", "1.jpg"))),
    ]
)
def test_recognition_model_resnet(image_input):

    predictor = ImageClassification()
    predictor.setModelTypeAsResNet50()
    predictor.setModelPath(os.path.join(test_folder, "data-models", "resnet50-19c8e357.pth"))
    predictor.loadModel()
    predictions, probabilities = predictor.classifyImage(image_input=image_input)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.parametrize(
    "image_input",
    [
        (os.path.join(test_folder, test_folder, "data-images", "1.jpg")),
        (cv2.imread(os.path.join(test_folder, test_folder, "data-images", "1.jpg"))),
        (Image.open(os.path.join(test_folder, test_folder, "data-images", "1.jpg"))),
    ]
)
def test_recognition_model_inceptionv3(image_input):

    predictor = ImageClassification()
    predictor.setModelTypeAsInceptionV3()
    predictor.setModelPath(os.path.join(test_folder, "data-models", "inception_v3_google-1a9a5a14.pth"))
    predictor.loadModel()
    predictions, probabilities = predictor.classifyImage(image_input=image_input)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.parametrize(
    "image_input",
    [
        (os.path.join(test_folder, test_folder, "data-images", "1.jpg")),
        (cv2.imread(os.path.join(test_folder, test_folder, "data-images", "1.jpg"))),
        (Image.open(os.path.join(test_folder, test_folder, "data-images", "1.jpg"))),
    ]
)
def test_recognition_model_densenet(image_input):

    predictor = ImageClassification()
    predictor.setModelTypeAsDenseNet121()
    predictor.setModelPath(os.path.join(test_folder, "data-models", "densenet121-a639ec97.pth"))
    predictor.loadModel()
    predictions, probabilities = predictor.classifyImage(image_input=image_input)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)