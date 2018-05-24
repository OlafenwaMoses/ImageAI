from imageai.Detection import ObjectDetection
import os
from time import time

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

our_time = time()
custom_objects = detector.CustomObjects(car=True, motorcycle=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3custom.jpg"))
print("IT TOOK : ", time() - our_time)
for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
    print("--------------------------------")
