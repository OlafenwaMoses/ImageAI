from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.evaluateModel(model_path="hololens/models", json_path="hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)



"""
SAMPLE RESULT


Model File:  hololens/models/detection_model-ex-07--loss-4.42.h5 

Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9231
mAP: 0.9231
===============================
Model File:  hololens/models/detection_model-ex-10--loss-3.95.h5 

Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9725
mAP: 0.9725
===============================
Model File:  hololens/models/detection_model-ex-05--loss-5.26.h5 

Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9204
mAP: 0.9204
===============================
Model File:  hololens/models/detection_model-ex-03--loss-6.44.h5 

Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.8120
mAP: 0.8120
===============================
Model File:  hololens/models/detection_model-ex-18--loss-2.96.h5 

Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9431
mAP: 0.9431
===============================
Model File:  hololens/models/detection_model-ex-17--loss-3.10.h5 

Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9404
mAP: 0.9404
===============================
Model File:  hololens/models/detection_model-ex-08--loss-4.16.h5 

Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9725
mAP: 0.9725
===============================
"""