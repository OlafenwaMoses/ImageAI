from imageai.Detection import ObjectDetection
import os
from time import time

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
detector.loadModel()

our_time = time()

custom = detector.CustomObjects(bicycle=True, backpack=True)

detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_image=os.path.join(execution_path , "7.jpg"), output_image_path=os.path.join(execution_path , "7-detected.jpg"), minimum_percentage_probability=40)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"]  )
    print("--------------------------------")
