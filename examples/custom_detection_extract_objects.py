from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolov3_hololens-yolo_mAP-0.82726_epoch-73.pt") # https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3_hololens-yolo_mAP-0.82726_epoch-73.pt
detector.setJsonPath("hololens-yolo_yolov3_detection_config.json") # https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/hololens-yolo_yolov3_detection_config.json
detector.loadModel()
detections, extracted_objects_array = detector.detectObjectsFromImage(input_image="holo2.jpg", output_image_path="holo2-detected.jpg", extract_detected_objects=True)

for detection, object_path in zip(detections, extracted_objects_array):
    print(object_path)
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("---------------")

"""
SAMPLE RESULT

holo2-detected-objects\hololens-1.jpg
hololens  :  39.69653248786926  :  [611, 74, 751, 154]
---------------

holo2-detected-objects\hololens-1.jpg
hololens  :  87.6643180847168  :  [23, 46, 90, 79]
---------------

holo2-detected-objects\hololens-1.jpg
hololens  :  89.25175070762634  :  [191, 66, 243, 95]
---------------

holo2-detected-objects\hololens-1.jpg
hololens  :  64.49641585350037  :  [437, 81, 514, 133]
---------------

holo2-detected-objects\hololens-1.jpg
hololens  :  91.78624749183655  :  [380, 113, 423, 138]
---------------
"""