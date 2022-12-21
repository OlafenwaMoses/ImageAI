# ImageAI : Custom Detection Model Training 

---

**ImageAI** provides the most simple and powerful approach to training custom object detection models
using the YOLOv3 architeture, which
which you can load into the `imageai.Detection.Custom.CustomObjectDetection` class. This allows
 you to train your own model on any set of images that corresponds to any type of objects of interest.
The training process generates a JSON file that maps the objects names in your image dataset and the detection anchors, as well as creates lots of models. In choosing the best model for your custom object detection task, an `evaluateModel()` function has been provided to compute the **mAP** of your saved models by allowing you to state your desired **IoU** and **Non-maximum Suppression** values. Then you can perform custom
object detection using the model and the JSON file generated. 

### TABLE OF CONTENTS
- <a href="#preparingdataset" > :white_square_button: Preparing your custom dataset</a>
- <a href="#trainingdataset" > :white_square_button: Training on your custom Dataset</a>
- <a href="#evaluatingmodels" > :white_square_button: Evaluating your saved detection models' mAP</a>


### Preparing your custom dataset
<div id="preparingdataset"></div>

To train a custom detection model, you need to prepare the images you want to use to train the model. 
You will prepare the images as follows: 

1. Decide the type of object(s) you want to detect and collect about **200 (minimum recommendation)** or more picture of each of the object(s)
2. Once you have collected the images, you need to annotate the object(s) in the images. **ImageAI** uses the **Pascal VOC format** for image annotation. You can generate this annotation for your images using the easy to use [**LabelImg**](https://github.com/tzutalin/labelImg) image annotation tool, available for Windows, Linux and MacOS systems. Open the link below to install the annotation tool. See: [https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)
3. When you are done annotating your images, **annotation XML** files will be generated for each image in your dataset. The **annotation XML** file describes each or **all** of the objects in the image. For example,  if each image your image names are **image(1).jpg**, **image(2).jpg**, **image(3).jpg** till **image(z).jpg**; the corresponding annotation for each of the images will be **image(1).xml**, **image(2).xml**, **image(3).xml** till **image(z).xml**. 
4. Once you have the annotations for all your images, create a folder for your dataset (E.g headsets) and in this parent folder, create child folders **train** and **validation**
5. In the train folder, create **images** and **annotations**
 sub-folders. Put about 70-80% of your dataset of each object's images in the **images** folder and put the corresponding annotations for these images in the **annotations** folder.  
6. In the validation folder, create **images** and **annotations** sub-folders. Put the rest of your dataset images in the **images** folder and put the corresponding annotations for these images in the **annotations** folder.
7. Once you have done this, the structure of your image dataset folder should look like below: 
    ```
    >> train    >> images       >> img_1.jpg  (shows Object_1)
                >> images       >> img_2.jpg  (shows Object_2)
                >> images       >> img_3.jpg  (shows Object_1, Object_3 and Object_n)
                >> annotations  >> img_1.xml  (describes Object_1)
                >> annotations  >> img_2.xml  (describes Object_2)
                >> annotations  >> img_3.xml  (describes Object_1, Object_3 and Object_n)
    
    >> validation   >> images       >> img_151.jpg (shows Object_1, Object_3 and Object_n)
                    >> images       >> img_152.jpg (shows Object_2)
                    >> images       >> img_153.jpg (shows Object_1)
                    >> annotations  >> img_151.xml (describes Object_1, Object_3 and Object_n)
                    >> annotations  >> img_152.xml (describes Object_2)
                    >> annotations  >> img_153.xml (describes Object_1)
     ```
8. You can train your custom detection model completely from scratch or use transfer learning (recommended for better accuracy) from a pre-trained YOLOv3 model. Also, we have provided a sample annotated Hololens and Headsets (Hololens and Oculus) dataset for you to train with. Download the pre-trained YOLOv3 model and the sample datasets in the link below.  

[https://github.com/OlafenwaMoses/ImageAI/releases/tag/essential-v4](https://github.com/OlafenwaMoses/ImageAI/releases/tag/essential-v4)


### Training on your custom dataset
<div id="trainingdataset"></div>

Before you start training your custom detection model, kindly take note of the following: 

- The default **batch_size** is 4. If you are training with **Google Colab**, this will be fine. However, I will advice you use a more powerful GPU than the K80 offered by Colab as the higher your **batch_size (8, 16)**, the better the accuracy of your detection model. 
- If you experience <i>'_TfDeviceCaptureOp' object has no attribute '_set_device_from_string'</i> error in Google Colab, it is due to a bug in **Tensorflow**. You can solve this by installing **Tensorflow GPU 1.13.1**. 
    ```bash
     pip3 install tensorflow-gpu==1.13.1
    ```

Then your training code goes as follows: 
```python
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
# In the above,when training for detecting multiple objects,
#set object_names_array=["object1", "object2", "object3",..."objectz"]
trainer.trainModel()
```

 Yes! Just 6 lines of code and you can train object detection models on your custom dataset.
Now lets take a look at how the code above works. 

```python
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
```

In the first line, we import the **ImageAI** detection model training class, then we define the model trainer in the second line,
 we set the network type in the third line and set the path to the image dataset we want to train the network on.

```python
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
```


In the line above, we configured our detection model trainer. The parameters we stated in the function as as below:  

- **num_objects** : this is an array containing the names of the objects in our dataset
- **batch_size** : this is to state the batch size for the training
- **num_experiments** : this is to state the number of times the network will train over all the training images,
 which is also called epochs 
- **train_from_pretrained_model(optional)** : this is to train using transfer learning from a pre-trained **YOLOv3** model

```python
trainer.trainModel()
```


When you start the training, you should see something like this in the console: 
```
Using TensorFlow backend.
Generating anchor boxes for training images and annotation...
Average IOU for 9 anchors: 0.78
Anchor Boxes generated.
Detection configuration saved in  hololens/json/detection_config.json
Training on: 	['hololens']
Training with Batch Size:  4
Number of Experiments:  200

Epoch 1/200
480/480 [==============================] - 395s 823ms/step - loss: 36.9000 - yolo_layer_1_loss: 3.2970 - yolo_layer_2_loss: 9.4923 - yolo_layer_3_loss: 24.1107 - val_loss: 15.6321 - val_yolo_layer_1_loss: 2.0275 - val_yolo_layer_2_loss: 6.4191 - val_yolo_layer_3_loss: 7.1856
Epoch 2/200
480/480 [==============================] - 293s 610ms/step - loss: 11.9330 - yolo_layer_1_loss: 1.3968 - yolo_layer_2_loss: 4.2894 - yolo_layer_3_loss: 6.2468 - val_loss: 7.9868 - val_yolo_layer_1_loss: 1.7054 - val_yolo_layer_2_loss: 2.9156 - val_yolo_layer_3_loss: 3.3657
Epoch 3/200
480/480 [==============================] - 293s 610ms/step - loss: 7.1228 - yolo_layer_1_loss: 1.0583 - yolo_layer_2_loss: 2.2863 - yolo_layer_3_loss: 3.7782 - val_loss: 6.4964 - val_yolo_layer_1_loss: 1.1391 - val_yolo_layer_2_loss: 2.2058 - val_yolo_layer_3_loss: 3.1514
Epoch 4/200
480/480 [==============================] - 297s 618ms/step - loss: 5.5802 - yolo_layer_1_loss: 0.9742 - yolo_layer_2_loss: 1.8916 - yolo_layer_3_loss: 2.7144 - val_loss: 6.4275 - val_yolo_layer_1_loss: 1.6153 - val_yolo_layer_2_loss: 2.1203 - val_yolo_layer_3_loss: 2.6919
Epoch 5/200
480/480 [==============================] - 295s 615ms/step - loss: 4.8717 - yolo_layer_1_loss: 0.7568 - yolo_layer_2_loss: 1.6641 - yolo_layer_3_loss: 2.4508 - val_loss: 6.3723 - val_yolo_layer_1_loss: 1.6434 - val_yolo_layer_2_loss: 2.1188 - val_yolo_layer_3_loss: 2.6101
Epoch 6/200
480/480 [==============================] - 300s 624ms/step - loss: 4.7989 - yolo_layer_1_loss: 0.8708 - yolo_layer_2_loss: 1.6683 - yolo_layer_3_loss: 2.2598 - val_loss: 5.8672 - val_yolo_layer_1_loss: 1.2349 - val_yolo_layer_2_loss: 2.0504 - val_yolo_layer_3_loss: 2.5820
Epoch 7/200
```

Let us explain the details shown above: 
```
Using TensorFlow backend.
Generating anchor boxes for training images and annotation...
Average IOU for 9 anchors: 0.78
Anchor Boxes generated.
Detection configuration saved in  hololens/json/detection_config.json
Training on: 	['hololens']
Training with Batch Size:  4
Number of Experiments:  200
```

The above details signifies the following: 
- **ImageAI** autogenerates the best match detection **anchor boxes** for your image dataset. 

- The anchor boxes and the object names mapping are saved in 
**json/detection_config.json** path of in the image dataset folder. Please note that for every new training you start, a new **detection_config.json** file is generated and is only compatible with the model saved during that training.

```
Epoch 1/200
480/480 [==============================] - 395s 823ms/step - loss: 36.9000 - yolo_layer_1_loss: 3.2970 - yolo_layer_2_loss: 9.4923 - yolo_layer_3_loss: 24.1107 - val_loss: 15.6321 - val_yolo_layer_1_loss: 2.0275 - val_yolo_layer_2_loss: 6.4191 - val_yolo_layer_3_loss: 7.1856
Epoch 2/200
480/480 [==============================] - 293s 610ms/step - loss: 11.9330 - yolo_layer_1_loss: 1.3968 - yolo_layer_2_loss: 4.2894 - yolo_layer_3_loss: 6.2468 - val_loss: 7.9868 - val_yolo_layer_1_loss: 1.7054 - val_yolo_layer_2_loss: 2.9156 - val_yolo_layer_3_loss: 3.3657
Epoch 3/200
480/480 [==============================] - 293s 610ms/step - loss: 7.1228 - yolo_layer_1_loss: 1.0583 - yolo_layer_2_loss: 2.2863 - yolo_layer_3_loss: 3.7782 - val_loss: 6.4964 - val_yolo_layer_1_loss: 1.1391 - val_yolo_layer_2_loss: 2.2058 - val_yolo_layer_3_loss: 3.1514
Epoch 4/200
480/480 [==============================] - 297s 618ms/step - loss: 5.5802 - yolo_layer_1_loss: 0.9742 - yolo_layer_2_loss: 1.8916 - yolo_layer_3_loss: 2.7144 - val_loss: 6.4275 - val_yolo_layer_1_loss: 1.6153 - val_yolo_layer_2_loss: 2.1203 - val_yolo_layer_3_loss: 2.6919
Epoch 5/200
480/480 [==============================] - 295s 615ms/step - loss: 4.8717 - yolo_layer_1_loss: 0.7568 - yolo_layer_2_loss: 1.6641 - yolo_layer_3_loss: 2.4508 - val_loss: 6.3723 - val_yolo_layer_1_loss: 1.6434 - val_yolo_layer_2_loss: 2.1188 - val_yolo_layer_3_loss: 2.6101
Epoch 6/200
480/480 [==============================] - 300s 624ms/step - loss: 4.7989 - yolo_layer_1_loss: 0.8708 - yolo_layer_2_loss: 1.6683 - yolo_layer_3_loss: 2.2598 - val_loss: 5.8672 - val_yolo_layer_1_loss: 1.2349 - val_yolo_layer_2_loss: 2.0504 - val_yolo_layer_3_loss: 2.5820
Epoch 7/200
```

- The above signifies the progress of the training. 
- For each experiment (Epoch), the general  total validation loss (E.g - loss: 4.7582) is reported. 
- For each drop in the loss after an experiment, a model is saved in the **hololens/models** folder. The lower the loss, the better the model. 
- **Tensorboard** report file for the training will be saved in the **hololens/logs** folder.

Once you are done training, you can visit the link below for performing object detection with your **custom detection model** and **detection_config.json** file.

[Detection/Custom/CUSTOMDETECTION.md](./CUSTOMDETECTION.md)
 
 
 
### Evaluating your saved detection models' mAP
 <div id="evaluatingmodels"></div>

After training on your custom dataset, you can evaluate the mAP of your saved models by specifying your desired IoU and Non-maximum suppression values. See details as below:

- **Single Model Evaluation:** To evaluate a single model, simply use the example code below with the path to your dataset directory, the model file and the **detection_config.json** file saved during the training. In the example, we used an **object_threshold** of 0.3 ( percentage_score >= 30% ), **IoU** of 0.5 and **Non-maximum suppression** value of 0.5.
    ```python
    from imageai.Detection.Custom import DetectionModelTrainer
    
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="hololens")
    metrics = trainer.evaluateModel(model_path="detection_model-ex-60--loss-2.76.h5", json_path="detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
    ```
    Consider that `trainer.evaluateModel` method will show the metrics on standard output as shown below, 
    but also returns a list of dicts containing all the information that is displayed. 
    
    Sample Result:
    ```
    Model File:  hololens_detection_model-ex-09--loss-4.01.h5 
    Using IoU :  0.5
    Using Object Threshold :  0.3
    Using Non-Maximum Suppression :  0.5
    hololens: 0.9613
    mAP: 0.9613
    ===============================
    ```
    Let's see how those metrics looks like:
    ```
    [{
        'average_precision': {'hololens': 0.9613334437735249},
        'map': 0.9613334437735249,
        'model_file': 'hololens_detection_model-ex-09--loss-4.01.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    }]
    ```
- **Multi Model Evaluation:** To evaluate all your saved models, simply parse in the path to the folder containing the models as the **model_path** as seen in the example below:
    ```python
    from imageai.Detection.Custom import DetectionModelTrainer
    
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="hololens")
    metrics = trainer.evaluateModel(model_path="hololens/models", json_path="hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
    ```
    Sample Result:
    ```
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
    ```
    Let's see how those metrics looks like:
    ```
    [{
        'average_precision': {'hololens': 0.9231334437735249},
        'map': 0.9231334437735249,
        'model_file': 'hololens/models/detection_model-ex-07--loss-4.42.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.9725334437735249},
        'map': 0.97251334437735249,
        'model_file': 'hololens/models/detection_model-ex-10--loss-3.95.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.92041334437735249},
        'map': 0.92041334437735249,
        'model_file': 'hololens/models/detection_model-ex-05--loss-5.26.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.81201334437735249},
        'map': 0.81201334437735249,
        'model_file': 'hololens/models/detection_model-ex-03--loss-6.44.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.94311334437735249},
        'map': 0.94311334437735249,
        'model_file': 'hololens/models/detection_model-ex-18--loss-2.96.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.94041334437735249},
        'map': 0.94041334437735249,
        'model_file': 'hololens/models/detection_model-ex-17--loss-3.10.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.97251334437735249},
        'map': 0.97251334437735249,
        'model_file': 'hololens/models/detection_model-ex-08--loss-4.16.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    }
    ]
    ```


###  >> Documentation
<div id="documentation" ></div>

We have provided full documentation for all **ImageAI** classes and functions in 3 major languages. Find links below: 

* Documentation - **English Version**  [https://imageai.readthedocs.io](https://imageai.readthedocs.io) 
* Documentation - **Chinese Version**  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)
* Documentation - **French Version**  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)






