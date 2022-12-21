from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="yolov3.pt")
#download pre-trained model via https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt
# If you are training to detect more than 1 object, set names of objects above like object_names_array=["hololens", "google-glass", "oculus", "magic-leap"]
trainer.trainModel()



"""
SAMPLE RESULT

Generating anchor boxes for training images...
thr=0.25: 1.0000 best possible recall, 6.93 anchors past thr
n=9, img_size=416, metric_all=0.463/0.856-mean/best, past_thr=0.549-mean:
====================
Pretrained YOLOv3 model loaded to initialize weights
====================
Epoch 1/100
----------
Train:
30it [00:14,  2.09it/s]
    box loss-> 0.09820, object loss-> 0.27985, class loss-> 0.00000
Validation:
15it [01:45,  7.05s/it]
    recall: 0.085714 precision: 0.000364 mAP@0.5: 0.000186, mAP@0.5-0.95: 0.000030

Epoch 2/100
----------
Train:
30it [00:07,  4.25it/s]
    box loss-> 0.08691, object loss-> 0.07011, class loss-> 0.00000
Validation:
15it [01:37,  6.53s/it]
    recall: 0.214286 precision: 0.000854 mAP@0.5: 0.000516, mAP@0.5-0.95: 0.000111
"""

