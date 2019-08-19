from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
#download pre-trained model via https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5
# If you are training to detect more than 1 object, set names of objects above like object_names_array=["hololens", "google-glass", "oculus", "magic-leap"]
trainer.trainModel()



"""
SAMPLE RESULT

Using TensorFlow backend.
Generating anchor boxes for training images and annotation...
Average IOU for 9 anchors: 0.78
Anchor Boxes generated.
Detection configuration saved in  hololens/json/detection_config.json
Training on: 	['hololens']
Training with Batch Size:  4
Number of Experiments:  200



Epoch 1/200
 - 733s - loss: 34.8253 - yolo_layer_1_loss: 6.0920 - yolo_layer_2_loss: 11.1064 - yolo_layer_3_loss: 17.6269 - val_loss: 20.5028 - val_yolo_layer_1_loss: 4.0171 - val_yolo_layer_2_loss: 7.5175 - val_yolo_layer_3_loss: 8.9683
Epoch 2/200
 - 648s - loss: 11.1396 - yolo_layer_1_loss: 2.1209 - yolo_layer_2_loss: 4.0063 - yolo_layer_3_loss: 5.0124 - val_loss: 7.6188 - val_yolo_layer_1_loss: 1.8513 - val_yolo_layer_2_loss: 2.2446 - val_yolo_layer_3_loss: 3.5229
Epoch 3/200
 - 674s - loss: 6.4360 - yolo_layer_1_loss: 1.3500 - yolo_layer_2_loss: 2.2343 - yolo_layer_3_loss: 2.8518 - val_loss: 7.2326 - val_yolo_layer_1_loss: 1.8762 - val_yolo_layer_2_loss: 2.3802 - val_yolo_layer_3_loss: 2.9762
Epoch 4/200
 - 634s - loss: 5.3801 - yolo_layer_1_loss: 1.0323 - yolo_layer_2_loss: 1.7854 - yolo_layer_3_loss: 2.5624 - val_loss: 6.3730 - val_yolo_layer_1_loss: 1.4272 - val_yolo_layer_2_loss: 2.0534 - val_yolo_layer_3_loss: 2.8924
Epoch 5/200
 - 645s - loss: 5.2569 - yolo_layer_1_loss: 0.9953 - yolo_layer_2_loss: 1.8611 - yolo_layer_3_loss: 2.4005 - val_loss: 6.0458 - val_yolo_layer_1_loss: 1.7037 - val_yolo_layer_2_loss: 1.9754 - val_yolo_layer_3_loss: 2.3667
Epoch 6/200
 - 655s - loss: 4.7582 - yolo_layer_1_loss: 0.9959 - yolo_layer_2_loss: 1.5986 - yolo_layer_3_loss: 2.1637 - val_loss: 5.8313 - val_yolo_layer_1_loss: 1.1880 - val_yolo_layer_2_loss: 1.9962 - val_yolo_layer_3_loss: 2.6471
Epoch 7/200

"""

