from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.evaluateModel(model_path="hololens-ex-60--loss-2.76.h5", json_path="detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)

# download JSON file via https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/detection_config.json
# download detection model via https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/hololens-ex-60--loss-2.76.h5



"""
SAMPLE RESULT


Model File:  hololens_detection_model-ex-09--loss-4.01.h5 

Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9613
mAP: 0.9613
===============================
"""