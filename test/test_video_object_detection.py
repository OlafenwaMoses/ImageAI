from imageai.Detection import VideoObjectDetection
import cv2
import pytest
import os
from os.path import dirname
from numpy import ndarray
import keras

main_folder = os.getcwd()
video_file = os.path.join(main_folder, "data-videos", "traffic-micro.mp4")
video_file_output = os.path.join(main_folder, "data-temp", "traffic-micro-detected")



@pytest.fixture
def clear_keras_session():
    try:
        keras.backend.clear_session()
    except:
        None


@pytest.mark.detection
@pytest.mark.video_detection
@pytest.mark.retinanet
def test_video_detection_retinanet(clear_keras_session):


    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path=os.path.join(main_folder, "data-models", "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel(detection_speed="fastest")
    video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True)

    assert os.path.exists(video_file_output + ".avi")
    assert isinstance(video_path, str)
    os.remove(video_file_output + ".avi")



@pytest.mark.detection
@pytest.mark.video_detection
@pytest.mark.yolov3
def test_video_detection_yolov3(clear_keras_session):


    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path=os.path.join(main_folder, "data-models", "yolo.h5"))
    detector.loadModel(detection_speed="faster")
    video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True)

    assert os.path.exists(video_file_output + ".avi")
    assert isinstance(video_path, str)
    os.remove(video_file_output + ".avi")


@pytest.mark.detection
@pytest.mark.video_detection
@pytest.mark.tiny_yolov3
def test_video_detection_tiny_yolov3(clear_keras_session):

    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(model_path=os.path.join(main_folder, "data-models", "yolo-tiny.h5"))
    detector.loadModel(detection_speed="fast")
    video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True)

    assert os.path.exists(video_file_output + ".avi")
    assert isinstance(video_path, str)
    os.remove(video_file_output + ".avi")




@pytest.mark.detection
@pytest.mark.video_detection
@pytest.mark.retinanet
@pytest.mark.video_analysis
def test_video_detection_retinanet_analysis(clear_keras_session):

    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path=os.path.join(main_folder, "data-models", "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel(detection_speed="fastest")
    video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True, per_frame_function=forFrame, per_second_function=forSecond, return_detected_frame=True)

    assert os.path.exists(video_file_output + ".avi")
    assert isinstance(video_path, str)
    os.remove(video_file_output + ".avi")


def forFrame(frame_number, output_array, output_count, detected_frame):
    assert isinstance(detected_frame, ndarray)
    assert isinstance(frame_number, int)
    assert isinstance(output_array, list)
    assert isinstance(output_array[0], dict)
    assert isinstance(output_array[0]["name"], str)
    assert isinstance(output_array[0]["percentage_probability"], float)
    assert isinstance(output_array[0]["box_points"], list)

    assert isinstance(output_count, dict)
    for a_key in dict(output_count).keys():
        assert isinstance(a_key, str)
        assert isinstance(output_count[a_key], int)

def forSecond(second_number, output_arrays, count_arrays, average_output_count, detected_frame):
    assert isinstance(detected_frame, ndarray)
    assert isinstance(second_number, int)
    assert isinstance(output_arrays, list)
    assert isinstance(output_arrays[0], list)

    assert isinstance(output_arrays[0][0], dict)
    assert isinstance(output_arrays[0][0]["name"], str)
    assert isinstance(output_arrays[0][0]["percentage_probability"], float)
    assert isinstance(output_arrays[0][0]["box_points"], list)

    assert isinstance(count_arrays, list)
    assert isinstance(count_arrays[0], dict)
    for a_key in dict(count_arrays[0]).keys():
        assert isinstance(a_key, str)
        assert isinstance(count_arrays[0][a_key], int)

    assert isinstance(average_output_count, dict)
    for a_key2 in dict(average_output_count).keys():
        assert isinstance(a_key2, str)
        assert isinstance(average_output_count[a_key2], int)
