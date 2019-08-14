from imageai.Detection.Custom import CustomObjectDetection
import pytest
import os
import cv2
import shutil
from numpy import ndarray
import keras

main_folder = os.getcwd()

image_input = os.path.join(main_folder, "data-images", "14.jpg")
image_output = os.path.join(main_folder, "data-temp", "14-detected.jpg")
model_path = os.path.join(main_folder, "data-models", "hololens-ex-60--loss-2.76.h5")
model_json = os.path.join(main_folder, "data-json", "detection_config.json")



@pytest.mark.detection
@pytest.mark.yolov3
@pytest.mark.custom_detection
def test_object_detection_yolov3():
    try:
        keras.backend.clear_session()
    except:
        None
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.setJsonPath(model_json)
    detector.loadModel()
    results = detector.detectObjectsFromImage(input_image=image_input, output_image_path=image_output, minimum_percentage_probability=40)

    assert isinstance(results, list)
    for result in results:
        assert isinstance(result["name"], str)
        assert isinstance(result["percentage_probability"], float)
        assert isinstance(result["box_points"], list)
    assert os.path.exists(image_output)
    os.remove(image_output)

    results2, extracted_paths = detector.detectObjectsFromImage(input_image=image_input, output_image_path=image_output,
                                                                minimum_percentage_probability=30,
                                                                extract_detected_objects=True)

    assert isinstance(results2, list)
    assert isinstance(extracted_paths, list)
    assert os.path.isdir(os.path.join(image_output + "-objects"))
    for result2 in results2:
        assert isinstance(result2["name"], str)
        assert isinstance(result2["percentage_probability"], float)
        assert isinstance(result2["box_points"], list)

    for extracted_path in extracted_paths:
        assert os.path.exists(extracted_path)

    shutil.rmtree(os.path.join(image_output + "-objects"))



@pytest.mark.detection
@pytest.mark.yolov3
@pytest.mark.custom_detection
@pytest.mark.array_io
def test_object_detection_yolov3_array_io():
    try:
        keras.backend.clear_session()
    except:
        None

    image_input_array = cv2.imread(image_input)

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.setJsonPath(model_json)
    detector.loadModel()
    detected_array, results = detector.detectObjectsFromImage(input_image=image_input_array, input_type="array", minimum_percentage_probability=40, output_type="array")

    assert isinstance(detected_array, ndarray)
    assert isinstance(results, list)
    for result in results:
        assert isinstance(result["name"], str)
        assert isinstance(result["percentage_probability"], float)
        assert isinstance(result["box_points"], list)

    detected_array, results2, extracted_arrays = detector.detectObjectsFromImage(input_image=image_input, output_image_path=image_output, minimum_percentage_probability=40, extract_detected_objects=True, output_type="array")

    assert isinstance(results2, list)
    assert isinstance(extracted_arrays, list)
    for result2 in results2:
        assert isinstance(result2["name"], str)
        assert isinstance(result2["percentage_probability"], float)
        assert isinstance(result2["box_points"], list)

    for extracted_array in extracted_arrays:
        assert isinstance(extracted_array, ndarray)
