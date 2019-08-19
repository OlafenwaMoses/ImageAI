from imageai.Prediction import ImagePrediction
import os
import cv2
import pytest
from os.path import dirname
import keras

main_folder = os.getcwd()




@pytest.mark.squeezenet
@pytest.mark.recognition
def test_recognition_model_squeezenet():


    predictor = ImagePrediction()
    predictor.setModelTypeAsSqueezeNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "1.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.resnet
@pytest.mark.recognition
def test_recognition_model_resnet():

    predictor = ImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "1.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.inceptionv3
@pytest.mark.recognition
def test_recognition_model_inceptionv3():

    predictor = ImagePrediction()
    predictor.setModelTypeAsInceptionV3()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "1.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.densenet
@pytest.mark.recognition
def test_recognition_model_densenet():

    predictor = ImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "DenseNet-BC-121-32.h5"))
    predictor.loadModel()
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "1.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)


@pytest.mark.squeezenet
@pytest.mark.recognition
def test_recognition_model_squeezenet_array_input():


    predictor = ImagePrediction()
    predictor.setModelTypeAsSqueezeNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()
    image_array = cv2.imread(os.path.join(main_folder, main_folder, "data-images", "1.jpg"))
    predictions, probabilities = predictor.predictImage(image_input=image_array, input_type="array")

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.resnet
@pytest.mark.recognition
def test_recognition_model_resnet_array_input():

    predictor = ImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()
    image_array = cv2.imread(os.path.join(main_folder, main_folder, "data-images", "1.jpg"))
    predictions, probabilities = predictor.predictImage(image_input=image_array, input_type="array")

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.inceptionv3
@pytest.mark.recognition
def test_recognition_model_inceptionv3_array_input():

    predictor = ImagePrediction()
    predictor.setModelTypeAsInceptionV3()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()
    image_array = cv2.imread(os.path.join(main_folder, main_folder, "data-images", "1.jpg"))
    predictions, probabilities = predictor.predictImage(image_input=image_array, input_type="array")

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

@pytest.mark.densenet
@pytest.mark.recognition
def test_recognition_model_densenet_array_input():

    predictor = ImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "DenseNet-BC-121-32.h5"))
    predictor.loadModel()
    image_array = cv2.imread(os.path.join(main_folder, main_folder, "data-images", "1.jpg"))
    predictions, probabilities = predictor.predictImage(image_input=image_array, input_type="array")

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], float)

