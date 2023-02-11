# ImageAI : Custom Detection Model Training 

---

**ImageAI** provides the most simple and powerful approach to training custom object detection models
using the YOLOv3 architeture, which
which you can load into the `imageai.Detection.Custom.CustomObjectDetection` class. This allows
 you to train your own **YOLOv3** or **TinyYOLOv3** model on any set of images that corresponds to any type of objects of interest.
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
2. Once you have collected the images, you need to annotate the object(s) in the images. **ImageAI** uses the **YOLO** for image annotation. You can generate this annotation for your images using the easy to use [**LabelImg**](https://github.com/tzutalin/labelImg) image annotation tool, available for Windows, Linux and MacOS systems. Open the link below to install the annotation tool. See: [https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)
3. When you are done annotating your images, **annotation .txt** files will be generated for each image in your dataset. The **annotation .txt** file describes each or **all** of the objects in the image. For example,  if each image your image names are **image(1).jpg**, **image(2).jpg**, **image(3).jpg** till **image(z).jpg**; the corresponding annotation for each of the images will be **image(1).txt**, **image(2).txt**, **image(3).txt** till **image(z).txt**. 
4. Once you have the annotations for all your images, create a folder for your dataset (E.g headsets) and in this parent folder, create child folders **train** and **validation**
5. In the train folder, create **images** and **annotations**
 sub-folders. Put about 70-80% of your dataset of each object's images in the **images** folder and put the corresponding annotations for these images in the **annotations** folder.  
6. In the validation folder, create **images** and **annotations** sub-folders. Put the rest of your dataset images in the **images** folder and put the corresponding annotations for these images in the **annotations** folder.
7. Once you have done this, the structure of your image dataset folder should look like below: 
    ```
    >> train    >> images       >> img_1.jpg  (shows Object_1)
                >> images       >> img_2.jpg  (shows Object_2)
                >> images       >> img_3.jpg  (shows Object_1, Object_3 and Object_n)
                >> annotations  >> img_1.txt  (describes Object_1)
                >> annotations  >> img_2.txt  (describes Object_2)
                >> annotations  >> img_3.txt  (describes Object_1, Object_3 and Object_n)
    
    >> validation   >> images       >> img_151.jpg (shows Object_1, Object_3 and Object_n)
                    >> images       >> img_152.jpg (shows Object_2)
                    >> images       >> img_153.jpg (shows Object_1)
                    >> annotations  >> img_151.txt (describes Object_1, Object_3 and Object_n)
                    >> annotations  >> img_152.txt (describes Object_2)
                    >> annotations  >> img_153.txt (describes Object_1)
     ```
8. You can train your custom detection model completely from scratch or use transfer learning (recommended for better accuracy) from a pre-trained YOLOv3 model. Also, we have provided a sample annotated Hololens and Headsets (Hololens and Oculus) dataset for you to train with. Download the pre-trained YOLOv3 model and the sample datasets in the link below.  

Download dataset `hololens-yolo.zip` [here](https://github.com/OlafenwaMoses/ImageAI/releases/tag/test-resources-v3) and pre-trained model `yolov3.pt`  [here](https://github.com/OlafenwaMoses/ImageAI/releases/tag/3.0.0-pretrained)


### Training on your custom dataset
<div id="trainingdataset"></div>

Before you start training your custom detection model, kindly take note of the following: 

- The default **batch_size** is 4. If you are training with **Google Colab**, this will be fine. However, I will advice you use a more powerful GPU than the K80 offered by Colab as the higher your **batch_size (8, 16)**, the better the accuracy of your detection model. 

Then your training code goes as follows: 
```python
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens-yolo")
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="yolov3.pt")
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
trainer.setDataDirectory(data_directory="hololens-yolo")
```

In the first line, we import the **ImageAI** detection model training class, then we define the model trainer in the second line,
 we set the network type in the third line and set the path to the image dataset we want to train the network on.

```python
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="yolov3.pt")
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
.
.
.
.

```

Let us explain the details shown above: 
```
Generating anchor boxes for training images...
thr=0.25: 1.0000 best possible recall, 6.93 anchors past thr
n=9, img_size=416, metric_all=0.463/0.856-mean/best, past_thr=0.549-mean:
====================
Pretrained YOLOv3 model loaded to initialize weights
====================
```

The above details signifies the following: 
- **ImageAI** autogenerates the best match detection **anchor boxes** for your image dataset. 

- A the pretrained **yolov3.pt** was loaded to initalize the weights used to train the model.

```
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
```

- The above signifies the progress of the training. 
- For each experiment (Epoch), a number of metrics are computed. The important once fo chosing an accuate models is detailed below
  - The bounding box loss `box loss` is reported and expected to drop as the training progresses
  - The object localization loss  `object loss` is reported and expected to drop as the training progresses
  - The class loss  `class loss` is reported and expected to drop as the training progresses. If the class loss persists at 0.0000, it's because your dataset has a single class.
  - The `mAP50` and `mAP0.5-0.95` metrics are expected to increase. This signifies the models accuracy increases. There might be flunctuations in these metrics sometimes.
- For each increase in the `mAP50`  after an experiment, a model is saved in the **hololens-yolo/models** folder. The higher the mAP50, the better the model. 

Once you are done training, you can visit the link below for performing object detection with your **custom detection model** and **detection_config.json** file.

[Detection/Custom/CUSTOMDETECTION.md](./CUSTOMDETECTION.md)
 
 
###  >> Documentation
<div id="documentation" ></div>

We have provided full documentation for all **ImageAI** classes and functions. Find links below: 

* Documentation - **English Version**  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)






