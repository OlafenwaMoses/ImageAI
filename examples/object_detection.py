from imageai.Detection import ObjectDetection
import os
from time import time

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed="flash")

our_time = time()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "6.jpg"), output_image_path=os.path.join(execution_path , "6flash.jpg"), minimum_percentage_probability=30)
print("IT TOOK : ", time() - our_time)
for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
    print("--------------------------------")
