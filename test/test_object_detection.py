import os, sys
import cv2
from PIL import Image
import pytest
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(dirname(os.path.abspath(__file__)))))
from imageai.Detection import ObjectDetection

test_folder = dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "input_image",
    [
        (os.path.join(test_folder, test_folder, "data-images", "1.jpg"))
    ]
)
def test_object_detection_retinanet(input_image):
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(test_folder, "data-models", "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=input_image)

    assert type(detections) == list
    

    for eachObject in detections:
        assert type(eachObject) == dict
        assert "name" in eachObject.keys()
        assert type(eachObject["name"]) == str 
        assert "percentage_probability" in eachObject.keys()
        assert type(eachObject["percentage_probability"]) == float
        assert "box_points" in eachObject.keys()
        assert type(eachObject["box_points"]) == list
        box_points = eachObject["box_points"]
        for point in box_points:
            assert type(point) == int
        assert box_points[0] < box_points[2]
        assert box_points[1] < box_points[3]


@pytest.mark.parametrize(
    "input_image",
    [
        (os.path.join(test_folder, test_folder, "data-images", "1.jpg"))
    ]
)
def test_object_detection_yolov3(input_image):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(test_folder, "data-models", "yolov3.pt"))
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=input_image)

    assert type(detections) == list
    

    for eachObject in detections:
        assert type(eachObject) == dict
        assert "name" in eachObject.keys()
        assert type(eachObject["name"]) == str 
        assert "percentage_probability" in eachObject.keys()
        assert type(eachObject["percentage_probability"]) == float
        assert "box_points" in eachObject.keys()
        assert type(eachObject["box_points"]) == list
        box_points = eachObject["box_points"]
        for point in box_points:
            assert type(point) == int
        assert box_points[0] < box_points[2]
        assert box_points[1] < box_points[3]

@pytest.mark.parametrize(
    "input_image",
    [
        (os.path.join(test_folder, test_folder, "data-images", "1.jpg"))
    ]
)
def test_object_detection_tiny_yolov3(input_image):
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(test_folder, "data-models", "tiny-yolov3.pt"))
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=input_image)

    assert type(detections) == list
    

    for eachObject in detections:
        assert type(eachObject) == dict
        assert "name" in eachObject.keys()
        assert type(eachObject["name"]) == str 
        assert "percentage_probability" in eachObject.keys()
        assert type(eachObject["percentage_probability"]) == float
        assert "box_points" in eachObject.keys()
        assert type(eachObject["box_points"]) == list
        box_points = eachObject["box_points"]
        for point in box_points:
            assert type(point) == int
        assert box_points[0] < box_points[2]
        assert box_points[1] < box_points[3]


@pytest.mark.parametrize(
    "input_image",
    [
        (os.path.join(test_folder, test_folder, "data-images", "11.jpg"))
    ]
)
def test_object_detection_retinanet_custom_objects(input_image):
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(test_folder, "data-models", "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()

    custom = detector.CustomObjects(person=True, cell_phone=True)

    custom_detections = detector.detectObjectsFromImage(input_image=input_image, custom_objects=custom)
    
    for custom_detection in custom_detections:
        assert custom_detection["name"] in ["person", "cell phone"]

    detections = detector.detectObjectsFromImage(input_image=input_image)

    assert len(detections) > len(custom_detections)


@pytest.mark.parametrize(
    "input_image",
    [
        (os.path.join(test_folder, test_folder, "data-images", "11.jpg"))
    ]
)
def test_object_detection_yolov3_custom_objects(input_image):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(test_folder, "data-models", "yolov3.pt"))
    detector.loadModel()

    custom = detector.CustomObjects(person=True, cell_phone=True)

    custom_detections = detector.detectObjectsFromImage(input_image=input_image, custom_objects=custom)
    
    for custom_detection in custom_detections:
        assert custom_detection["name"] in ["person", "cell phone"]

    detections = detector.detectObjectsFromImage(input_image=input_image)

    assert len(detections) > len(custom_detections)


@pytest.mark.parametrize(
    "input_image",
    [
        (os.path.join(test_folder, test_folder, "data-images", "11.jpg"))
    ]
)
def test_object_detection_tiny_yolov3_custom_objects(input_image):
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(test_folder, "data-models", "tiny-yolov3.pt"))
    detector.loadModel()

    custom = detector.CustomObjects(person=True, cell_phone=True)

    custom_detections = detector.detectObjectsFromImage(input_image=input_image, custom_objects=custom)
    
    for custom_detection in custom_detections:
        assert custom_detection["name"] in ["person", "cell phone"]

    detections = detector.detectObjectsFromImage(input_image=input_image)

    assert len(detections) > len(custom_detections)

