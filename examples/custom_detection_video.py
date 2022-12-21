from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("yolov3_hololens-yolo_mAP-0.82726_epoch-73.pt") # https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3_hololens-yolo_mAP-0.82726_epoch-73.pt
video_detector.setJsonPath("hololens-yolo_yolov3_detection_config.json") # https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/hololens-yolo_yolov3_detection_config.json
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="holo1.mp4",
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,
                                          minimum_percentage_probability=40,
                                          log_progress=True)