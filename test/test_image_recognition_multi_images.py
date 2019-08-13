from imageai.Prediction import ImagePrediction
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


@pytest.mark.squeezenet
@pytest.mark.recognition
@pytest.mark.recognition_multi
def test_recognition_model_squeezenet():

    try:
        keras.backend.clear_session()
    except:
        None

    predictor = ImagePrediction()
    predictor.setModelTypeAsSqueezeNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()

    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], float)



@pytest.mark.resnet
@pytest.mark.recognition
@pytest.mark.recognition_multi
def test_recognition_model_resnet():
    try:
        keras.backend.clear_session()
    except:
        None
    predictor = ImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()

    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], float)

@pytest.mark.inceptionv3
@pytest.mark.recognition
@pytest.mark.recognition_multi
def test_recognition_model_inceptionv3():
    try:
        keras.backend.clear_session()
    except:
        None
    predictor = ImagePrediction()
    predictor.setModelTypeAsInceptionV3()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()

    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], float)

@pytest.mark.densenet
@pytest.mark.recognition
@pytest.mark.recognition_multi
def test_recognition_model_densenet():
    try:
        keras.backend.clear_session()
    except:
        None
    predictor = ImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "DenseNet-BC-121-32.h5"))
    predictor.loadModel()

    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], float)