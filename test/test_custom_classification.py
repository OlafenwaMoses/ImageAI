import os, sys
import cv2
from PIL import Image
import pytest
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(dirname(os.path.abspath(__file__)))))
from imageai.Classification.Custom import CustomImageClassification

test_folder = dirname(os.path.abspath(__file__))



@pytest.mark.parametrize(
    "image_input",
    [
        (os.path.join(test_folder, "data-images", "1.jpg")),
        (cv2.imread(os.path.join(test_folder, "data-images", "1.jpg"))),
        (Image.open(os.path.join(test_folder, "data-images", "1.jpg"))),
    ]
)
def test_recognition_model_mobilenetv2(image_input):

    classifier = CustomImageClassification()
    classifier.setModelTypeAsMobileNetV2()
    classifier.setModelPath(os.path.join(test_folder, "data-models", "mobilenet_v2-idenprof-test_acc_0.85300_epoch-92.pt"))
    classifier.setJsonPath(os.path.join(test_folder, "data-json", "idenprof_model_classes.json"))
    classifier.loadModel()
    predictions, probabilities = classifier.classifyImage(image_input=image_input, result_count=5)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)


@pytest.mark.parametrize(
    "image_input",
    [
        (os.path.join(test_folder, "data-images", "1.jpg")),
        (cv2.imread(os.path.join(test_folder, "data-images", "1.jpg"))),
        (Image.open(os.path.join(test_folder, "data-images", "1.jpg"))),
    ]
)
def test_recognition_model_resnet(image_input):

    classifier = CustomImageClassification()
    classifier.setModelTypeAsResNet50()
    classifier.setModelPath(os.path.join(test_folder, "data-models", "resnet50-idenprof-test_acc_0.78200_epoch-91.pt"))
    classifier.setJsonPath(os.path.join(test_folder, "data-json", "idenprof_model_classes.json"))
    classifier.loadModel()
    predictions, probabilities = classifier.classifyImage(image_input=image_input, result_count=5)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.parametrize(
    "image_input",
    [
        (os.path.join(test_folder, "data-images", "1.jpg")),
        (cv2.imread(os.path.join(test_folder, "data-images", "1.jpg"))),
        (Image.open(os.path.join(test_folder, "data-images", "1.jpg"))),
    ]
)
def test_recognition_model_inceptionv3(image_input):

    classifier = CustomImageClassification()
    classifier.setModelTypeAsInceptionV3()
    classifier.setModelPath(os.path.join(test_folder, "data-models", "inception_v3-idenprof-test_acc_0.81050_epoch-92.pt"))
    classifier.setJsonPath(os.path.join(test_folder, "data-json", "idenprof_model_classes.json"))
    classifier.loadModel()
    predictions, probabilities = classifier.classifyImage(image_input=image_input, result_count=5)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.parametrize(
    "image_input",
    [
        (os.path.join(test_folder, "data-images", "1.jpg")),
        (cv2.imread(os.path.join(test_folder, "data-images", "1.jpg"))),
        (Image.open(os.path.join(test_folder, "data-images", "1.jpg"))),
    ]
)
def test_recognition_model_densenet(image_input):

    classifier = CustomImageClassification()
    classifier.setModelTypeAsDenseNet121()
    classifier.setModelPath(os.path.join(test_folder, "data-models", "densenet121-idenprof-test_acc_0.82550_epoch-95.pt"))
    classifier.setJsonPath(os.path.join(test_folder, "data-json", "idenprof_model_classes.json"))
    classifier.loadModel()
    predictions, probabilities = classifier.classifyImage(image_input=image_input, result_count=5)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)