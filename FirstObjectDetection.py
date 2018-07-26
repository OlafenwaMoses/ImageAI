from imageai.Detection import ObjectDetection
import os
import time

execution_path = os.getcwd()
model_retinanet = 'D:/data_public/ImageAI/retinanet/resnet50_coco_best_v2.0.1.h5'

detector = ObjectDetection()
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_retinanet)
detector.loadModel()

input_image_path = os.path.join(execution_path , 'images', "image2.jpg")
output_image_path = os.path.join(execution_path , "image2new.jpg")
print(input_image_path)
print(output_image_path)
t0 = time.time()
detections = detector.detectObjectsFromImage(input_image=input_image_path, output_image_path=output_image_path, minimum_percentage_probability=30)
print('time :', time.time() - t0)
for i in range(10):
    t0 = time.time()
    detections = detector.detectObjectsFromImage(input_image=input_image_path, output_image_path=output_image_path, minimum_percentage_probability=30)
    print('time :', time.time() - t0)
for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")

