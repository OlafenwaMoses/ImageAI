import os
from imageai.Detection import VideoObjectDetection

PATH_HERE = os.getcwd()
PATH_MODEL = os.path.join(PATH_HERE, "yolo.h5")
PATH_VIDEO_INPUT = os.path.join(PATH_HERE, "traffic.mp4")
PATH_VIDEO_DETECT = os.path.join(PATH_HERE, "traffic_detected_custom")


def main(path_model, path_video, path_detect):
    detector = VideoObjectDetection()
    detector.set_model_type_as_yolo_v3()
    detector.set_model_path(path_model)
    detector.load_model()

    custom = VideoObjectDetection.custom_objects(person=True, motorcycle=True, bus=True)

    video_path = detector.detect_custom_objects(custom_objects=custom, input_file_path=path_video,
                                                output_file_path=path_detect, frames_per_second=20,
                                                log_progress=True)
    print(video_path)


if __name__ == '__main__':
    main(path_model=PATH_MODEL, path_video=PATH_VIDEO_INPUT, path_detect=PATH_VIDEO_DETECT)
