from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolov3.pt")) # https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt
detector.loadModel()

custom = detector.CustomObjects(person=True, motorcycle=True, bus=True)

video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom, input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_detected_custom")
                                , frames_per_second=20, log_progress=True)
print(video_path)