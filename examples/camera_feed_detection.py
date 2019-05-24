"""
Running the Yolo detector from raw video stream on computer

To use the pre-trained model download Yolo.h5::

    wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5

"""

import os

import cv2

from imageai.Detection import VideoObjectDetection

PATH_HERE = os.path.abspath(os.getcwd())
PATH_HERE = os.path.expanduser('/home/jb/Desktop')
PATH_MODEL = os.path.join(PATH_HERE , 'yolo.h5')
PATH_OUTPUT = os.path.join(PATH_HERE, 'camera_detected_video')
VIDEO_SOURCE = cv2.VideoCapture(0)


def main(camera, path_model, path_output):
    detector = VideoObjectDetection()
    detector.set_model_type_as_yolo_v3()
    detector.set_model_path(path_model)
    detector.load_model()

    video_path = detector.detect_objects(
        camera_input=camera,
        output_file_path=path_output,
        frames_per_second=20,
        log_progress=True,
        minimum_percentage_probability=30
    )

    print(video_path)


if __name__ == '__main__':
    main(camera=VIDEO_SOURCE, path_model=PATH_MODEL, path_output=PATH_OUTPUT)
