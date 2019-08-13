from imageai.Prediction import ImagePrediction
import os
import pytest
from os.path import dirname


TEST_FOLDER = os.path.dirname(__file__)
all_images = os.listdir(os.path.join(TEST_FOLDER, "data-images"))
# Hint: Exit code 137 typically means the process is killed because it was running out of memory
# Hint: Check if you can optimize the memory usage in your app
# Hint: Max memory usage of this container is 4281565184
all_images_array = [os.path.join(TEST_FOLDER, "data-images", image)
                    for image in all_images[:2]]


@pytest.mark.squeezenet
@pytest.mark.recognition
@pytest.mark.recognition_multi
def test_recognition_model_squeezenet():

    predictor = ImagePrediction()
    predictor.setModelTypeAsSqueezeNet()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()

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
    predictor = ImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()

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
    predictor = ImagePrediction()
    predictor.setModelTypeAsInceptionV3()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
    predictor.loadModel()

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
    predictor = ImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "DenseNet-BC-121-32.h5"))
    predictor.loadModel()

    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], float)