from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "2.jpg"), output_image_path=os.path.join(execution_path , "2_detected.jpg"), minimum_percentage_probability=40)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")