# ImageAI : Custom Detection Model Training <br>
<hr>
<br>
<b>ImageAI</b> provides the most simple and powerful approach to training custom object detection models
using the YOLOv3 architeture, which
which you can load into the <b>imageai.Detection.Custom.CustomObjectDetection</b> class. This allows
 you to train your own model on any set of images that corresponds to any type of objects of interest.
The training process generates a JSON file that maps the objects names in your image dataset and the detection anchors, as well as creates lots of models. In choosing the best model for your custom object detection task, an <b>evaluateModel()</b> function has been provided to compute the <b>mAP</b> of your saved models by allowing you to state your desired <b>IoU</b> and <b>Non-maximum Suppression</b> values. Then you can perform custom
object detection using the model and the JSON file generated. <br><br>

<h3><b><u>TABLE OF CONTENTS</u></b></h3>
<a href="#preparingdataset" > &#9635 Preparing your custom dataset</a><br>
<a href="#trainingdataset" > &#9635 Training on your custom Dataset</a><br>
<a href="#evaluatingmodels" > &#9635 Evaluating your saved detection models' mAP</a><br>


<div id="preparingdataset"></div>
<h3><b><u>Preparing your custom dataset</u></b></h3>

To train a custom detection model, you need to prepare the images you want to use to train the model. 
You will prepare the images as follows: <br>

1. Decide the type of object(s) you want to detect and collect about <b>200 (minimum recommendation)</b> or more picture of each of the object(s)
2. Once you have collected the images, you need to annotate the object(s) in the images. <b>ImageAI</b> uses the <b>Pascal VOC format</b> for image annotation. You can generate this annotation for your images using the easy to use <a href="https://github.com/tzutalin/labelImg" ><b>LabelImg</b></a> image annotation tool, available for Windows, Linux and MacOS systems. Open the link below to install the annotation tool. <br><br>
<a href="https://github.com/tzutalin/labelImg" >https://github.com/tzutalin/labelImg</a>

3. When you are done annotating your images, <b>annotation XML</b> files will be generated for each image in your dataset. For example, if your image names are <b>image(1).jpg</b>, <b>image(2).jpg</b>, <b>image(3).jpg</b> till <b>image(z).jpg</b>; the corresponding annotation for each of the images will be <b>image(1).xml</b>, <b>image(2).xml</b>, <b>image(3).xml</b> till <b>image(z).xml</b>. 
4. Once you have the annotations for all your images, create a folder for your dataset (E.g headsets) and in this parent folder, create child folders <b>train</b> and <b>validation</b>
5. In the train folder, create <b>images</b> and <b>annotations</b>
 sub-folders. Put about 70-80% of your dataset images in the <b>images</b> folder and put the corresponding annotations for these images in the <b>annotations</b> folder.  <br>
6. In the validation folder, create <b>images</b> and <b>annotations</b> sub-folders. Put the rest of your dataset images in the <b>images</b> folder and put the corresponding annotations for these images in the <b>annotations</b> folder.  <br>
8. Once you have done this, the structure of your image dataset folder should look like below: <br> <br>

<pre>	>> train    >> images       >> img_1.jpg
                    >> images       >> img_2.jpg
                    >> images       >> img_3.jpg
                    >> annotations  >> img_1.xml
                    >> annotations  >> img_2.xml
                    >> annotations  >> img_3.xml


        >> validation   >> images       >> img_151.jpg
                        >> images       >> img_152.jpg
                        >> images       >> img_153.jpg
                        >> annotations  >> img_151.xml
                        >> annotations  >> img_152.xml
                        >> annotations  >> img_153.xml
     </pre>

9. You can train your custom detection model completely from scratch or use transfer learning (recommended for better accuracy) from a pre-trained YOLOv3 model. Also, we have provided a sample annotated Hololens and Headsets (Hololens and Oculus) dataset for you to train with. Download the pre-trained YOLOv3 model and the sample datasets in the link below. <br><br> 

<a href="https://github.com/OlafenwaMoses/ImageAI/releases/tag/essential-v4" >https://github.com/OlafenwaMoses/ImageAI/releases/tag/essential-v4</a>


<div id="trainingdataset"></div>
<h3><b><u>Training on your custom dataset</u></b></h3>
Before you start training your custom detection model, kindly take note of the following: <br>

- The default <b>batch_size</b> is 4. If you are training with <b>Google Colab</b>, this will be fine. However, I will advice you use a more powerful GPU than the K80 offered by Colab as the higher your <b>batch_size (8, 16)</b>, the better the accuracy of your detection model. <br>
 - If you experience <i>'_TfDeviceCaptureOp' object has no attribute '_set_device_from_string'</i> error in Google Colab, it is due to a bug in <b>Tensorflow</b>. You can solve this by installing <b>Tensorflow GPU 1.13.1</b>. <br>
 
 <pre>pip3 install tensorflow-gpu==1.13.1</pre>

Then your training code goes as follows: <br> <br>
<pre>from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()
</pre> <br><br>
 Yes! Just 6 lines of code and you can train object detection models on your custom dataset.
Now lets take a look at how the code above works. <br>
<pre>from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")</pre>
<br>
In the first line, we import the <b>ImageAI</b> detection model training class, then we define the model trainer in the second line,
 we set the network type in the third line and set the path to the image dataset we want to train the network on.

<pre>trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
</pre>
<br>

In the line above, we configured our detection model trainer. The parameters we stated in the function as as below:  <br>

- <b>num_objects</b> : this is an array containing the names of the objects in our dataset<br>
- <b>batch_size</b> : this is to state the batch size for the training<br>
- <b>num_experiments</b> : this is to state the number of times the network will train over all the training images,
 which is also called epochs <br>
- <b>train_from_pretrained_model(optional)</b> : this is to train using transfer learning from a pre-trained <b>YOLOv3</b> model<br>
 
 
<pre>trainer.trainModel()
</pre>
<br>



When you start the training, you should see something like this in the console: <br>
<pre>

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

</pre>

<br>
Let us explain the details shown above: <br>

<pre>
Using TensorFlow backend.
Generating anchor boxes for training images and annotation...
Average IOU for 9 anchors: 0.78
Anchor Boxes generated.
Detection configuration saved in  hololens/json/detection_config.json
Training on: 	['hololens']
Training with Batch Size:  4
Number of Experiments:  200
</pre>

The above details signifies the following: <br>
- <b>ImageAI</b> autogenerates the best match detection <b>anchor boxes</b> for your image dataset. <br>

- The anchor boxes and the object names mapping are saved in 
<b>json/detection_config.json</b> path of in the image dataset folder. Please note that for every new training you start, a new <b>detection_config.json</b> file is generated and is only compatible with the model saved during that training.<br> <br>

<pre>
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
</pre>

- The above signifies the progress of the training. <br>
- For each experiment (Epoch), the general  total validation loss (E.g - loss: 4.7582) is reported. <br>
- For each drop in the loss after an experiment, a model is saved in the <b>hololens/models</b> folder. The lower the loss, the better the model. <br><br>

Once you are done training, you can visit the link below for performing object detection with your <b>custom detection model</b> and <b>detection_config.json</b> file.

 <a href="https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/Custom/CUSTOMDETECTION.md" >Detection/Custom/CUSTOMDETECTION.md</a>
 
 <br>
 
 <div id="evaluatingmodels"></div>
<h3><b><u>Evaluating your saved detection models' mAP</u></b></h3>

After training on your custom dataset, you can evaluate the mAP of your saved models by specifying your desired IoU and Non-maximum suppression values. See details as below: <br>
- <b>Single Model Evaluation:</b> To evaluate a single model, simply use the example code below with the path to your dataset directory, the model file and the <b>detection_config.json</b> file saved during the training. In the example, we used an <b>object_threshold</b> of 0.3 ( percentage_score >= 30% ), <b>IoU</b> of 0.5 and <b>Non-maximum suppression</b> value of 0.5.<br>

<pre>
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.evaluateModel(model_path="detection_model-ex-60--loss-2.76.h5", json_path="detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
</pre>

Sample Result:

<pre>

Model File:  hololens_detection_model-ex-09--loss-4.01.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9613
mAP: 0.9613
===============================

</pre>


 - <b>Multi Model Evaluation:</b> To evaluate all your saved models, simply parse in the path to the folder containing the models as the <b>model_path</b> as seen in the example below:<br>


<pre>
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.evaluateModel(model_path="hololens/models", json_path="hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)


</pre>
Sample Result:
<pre>

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

</pre>

<br><br>






<div id="documentation" ></div>
<h3><b><u> >> Documentation</u></b></h3>
We have provided full documentation for all <b>ImageAI</b> classes and functions in 3 major languages. Find links below: <br>

<b> >> Documentation - English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)</b> <br>
<b> >> Documentation - Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)</b>
<br>
<b> >> Documentation - French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)</b>






