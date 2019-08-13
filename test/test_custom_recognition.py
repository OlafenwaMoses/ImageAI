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



@pytest.mark.resnet
@pytest.mark.recognition_custom
def test_custom_recognition_model_resnet():

    try:
        keras.backend.clear_session()
    except:
        None

    predictor = CustomImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_resnet.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], str)


@pytest.mark.resnet
@pytest.mark.recognition_custom
def test_custom_recognition_full_model_resnet():
    try:
        keras.backend.clear_session()
    except:
        None
    predictor = CustomImagePrediction()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_full_resnet_ex-001_acc-0.119792.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadFullModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], str)


@pytest.mark.densenet
@pytest.mark.recognition_custom
def test_custom_recognition_model_densenet():
    try:
        keras.backend.clear_session()
    except:
        None
    predictor = CustomImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_densenet-0.763500.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(main_folder, main_folder, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], str)




@pytest.mark.resnet
@pytest.mark.recognition_custom
@pytest.mark.recognition_multi
def test_custom_recognition_model_resnet_multi():
    try:
        keras.backend.clear_session()
    except:
        None
    predictor = CustomImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_resnet.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], str)


@pytest.mark.resnet
@pytest.mark.recognition_custom
@pytest.mark.recognition_multi
def test_custom_recognition_full_model_resnet_multi():
    try:
        keras.backend.clear_session()
    except:
        None
    predictor = CustomImagePrediction()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_full_resnet_ex-001_acc-0.119792.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadFullModel(num_objects=10)
    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], str)


@pytest.mark.densenet
@pytest.mark.recognition_custom
@pytest.mark.recognition_multi
def test_custom_recognition_model_densenet_multi():
    try:
        keras.backend.clear_session()
    except:
        None
    predictor = CustomImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(main_folder, "data-models", "idenprof_densenet-0.763500.h5"))
    predictor.setJsonPath(model_json=os.path.join(main_folder, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], str)