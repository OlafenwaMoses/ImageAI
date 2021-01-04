# ImageAI : Image Prediction
A **DeepQuest AI** project [https://deepquestai.com](https://deepquestai.com)

---

### TABLE OF CONTENTS
- <a href="#firstprediction" > :white_square_button: First Prediction</a>
- <a href="#predictionspeed" > :white_square_button: Prediction Speed</a>
- <a href="#inputtype" > :white_square_button: Image Input Types</a>
- <a href="#threadprediction" > :white_square_button: Prediction in MultiThreading</a>
- <a href="#documentation" > :white_square_button: Documentation</a>

ImageAI provides 4 different algorithms and model types to perform image prediction.
To perform image prediction on any picture, take the following simple steps.  The 4 algorithms provided for
 image prediction include **MobileNetV2**, **ResNet50**, **InceptionV3** and **DenseNet121**. Each of these
  algorithms have individual model files which you must use depending on the choice of your algorithm. To download the
   model file for your choice of algorithm, click on any of the links below:
   
- **[MobileNetV2](https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/mobilenet_v2.h5)** _(Size = 4.82 mb, fastest prediction time and moderate accuracy)_
- **[ResNet50](https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_imagenet_tf.2.0.h5)** by Microsoft Research _(Size = 98 mb, fast prediction time and high accuracy)_
 - **[InceptionV3](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/inception_v3_weights_tf_dim_ordering_tf_kernels.h5)** by Google Brain team _(Size = 91.6 mb, slow prediction time and higher accuracy)_
 - **[DenseNet121](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/DenseNet-BC-121-32.h5)** by Facebook AI Research _(Size = 31.6 mb, slower prediction time and highest accuracy)_

 Great! Once you have downloaded this model file, start a new python project, and then copy the model file to your project
     folder where your python files (.py files) will be . Download the image below, or take any image on your computer
 and copy it to your python project's folder. Then create a python file and give it a name; an example is `FirstPrediction.py`.
      Then write the code below into the python file:
      
### FirstPrediction.py
<div id="firstprediction" ></div>

```python
from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "1.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
```

Sample Result:
![](../../data-images/1.jpg)

```
convertible : 52.459555864334106
sports_car : 37.61284649372101
pickup : 3.1751200556755066
car_wheel : 1.817505806684494
minivan : 1.7487050965428352
```

The code above works as follows:
```python
from imageai.Classification import ImageClassification
import os
```
The code above imports the `ImageAI` library and the python `os` class.
```python
execution_path = os.getcwd()
```
The above line obtains the path to the folder that contains your python file (in this example, your FirstPrediction.py).

```python
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))
```
In the lines above, we created and instance of the `ImagePrediction()` class in the first line, then we set the model type of the prediction object to ResNet by caling the `.setModelTypeAsResNet50()` in the second line and then we set the model path of the prediction object to the path of the model file (`resnet50_imagenet_tf.2.0.h5`) we copied to the python file folder in the third line.

```python
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "1.jpg"), result_count=5 )
```

In the above line, we defined 2 variables to be equal to the function called to predict an image, which is the `.classifyImage()` function, into which we parsed the path to our image and also state the number of prediction results we want to have (values from 1 to 1000) parsing `result_count=5`. The `.classifyImage()` function will return 2 array objects with the first (**predictions**) being an array of predictions and the second (**percentage_probabilities**) being an array of the corresponding percentage probability for each prediction.

```python
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : " , eachProbability)
```
The above line obtains each object in the **predictions** array, and also obtains the corresponding percentage probability from the **percentage_probabilities**, and finally prints the result of both to console.


### Prediction Speed
<div id="predictionspeed"></div>

**ImageAI** now provides prediction speeds for all image prediction tasks. The prediction speeds allow you to reduce the time of prediction at a rate between 20% - 60%, and yet having just slight changes but accurate prediction results. The available prediction speeds are **"normal"**(default), **"fast"**, **"faster"** and **"fastest"**.
All you need to do is to state the speed mode you desire when loading the model as seen below.

```python
prediction.loadModel(prediction_speed="fast")
```

To observe the differences in the prediction speeds, look below for each speed applied to multiple prediction with time taken to predict and predictions given. The results below are obtained from predictions performed on a Windows 8 laptop with Intel Celeron N2820 CPU, with processor speed of 2.13GHz

**Prediction Speed = "normal" , Prediction Time = 5.9 seconds**
```
convertible : 52.459555864334106
sports_car : 37.61284649372101
pickup : 3.1751200556755066
car_wheel : 1.817505806684494
minivan : 1.7487050965428352
-----------------------
toilet_tissue : 13.99008333683014
jeep : 6.842949986457825
car_wheel : 6.71963095664978
seat_belt : 6.704962253570557
minivan : 5.861184373497963
-----------------------
bustard : 52.03368067741394
vulture : 20.936034619808197
crane : 10.620515048503876
kite : 10.20539253950119
white_stork : 1.6472270712256432
-----------------------
```

**Prediction Speed = "fast" , Prediction Time = 3.4 seconds**
```
sports_car : 55.5136501789093
pickup : 19.860029220581055
convertible : 17.88402795791626
tow_truck : 2.357563190162182
car_wheel : 1.8646160140633583
-----------------------
drum : 12.241223454475403
toilet_tissue : 10.96322312951088
car_wheel : 10.776633024215698
dial_telephone : 9.840480983257294
toilet_seat : 8.989936858415604
-----------------------
vulture : 52.81011462211609
bustard : 45.628002285957336
kite : 0.8065823465585709
goose : 0.3629807382822037
crane : 0.21266008261591196
-----------------------
```

**Prediction Speed = "faster" , Prediction Time = 2.7 seconds**
```
sports_car : 79.90474104881287
tow_truck : 9.751049429178238
convertible : 7.056044787168503
racer : 1.8735893070697784
car_wheel : 0.7379394955933094
-----------------------
oil_filter : 73.52778315544128
jeep : 11.926891654729843
reflex_camera : 7.9965077340602875
Polaroid_camera : 0.9798810817301273
barbell : 0.8661789819598198
-----------------------
vulture : 93.00530552864075
bustard : 6.636220961809158
kite : 0.15161558985710144
bald_eagle : 0.10513027664273977
crane : 0.05982434959150851
-----------------------
```

**Prediction Speed = "fastest" , Prediction Time = 2.2 seconds**
```
tow_truck : 62.5033438205719
sports_car : 31.26143217086792
racer : 2.2139860317111015
fire_engine : 1.7813067883253098
ambulance : 0.8790366351604462
-----------------------
reflex_camera : 94.00787949562073
racer : 2.345871739089489
jeep : 1.6016140580177307
oil_filter : 1.4121259562671185
lens_cap : 0.1283118617720902
-----------------------
kite : 98.5377550125122
vulture : 0.7469987496733665
bustard : 0.36855682265013456
bald_eagle : 0.2437378279864788
great_grey_owl : 0.0699841941241175
-----------------------
```

**PLEASE NOTE:**  When adjusting speed modes, it is best to use models that have higher accuracies like the DenseNet or InceptionV3 models, or use it in case scenarios where the images predicted are iconic.


### Image Input Types
<div id="inputtype"></div>

Previous version of **ImageAI** supported only file inputs and accepts file paths to an image for image prediction.
Now, **ImageAI** supports 3 input types which are **file path to image file**(default), **numpy array of image** and **image file stream**.
This means you can now perform image prediction in production applications such as on a web server and system
 that returns file in any of the above stated formats.

To perform image prediction with numpy array or file stream input, you just need to state the input type
in the `.classifyImage()` function. See example below.

```python
predictions, probabilities = prediction.classifyImage(image_array, result_count=5 , input_type="array" ) # For numpy array input type
predictions, probabilities = prediction.classifyImage(image_stream, result_count=5 , input_type="stream" ) # For file stream input type
```

### Prediction in MultiThreading
<div id="threadprediction"></div>

When developing programs that run heavy task on the deafult thread like User Interfaces (UI),
 you should consider running your predictions in a new thread. When running image prediction using ImageAI in
 a new thread, you must take note the following:
- You can create your prediction object, set its model type, set model path and json path
outside the new thread.
- The `.loadModel()` must be in the new thread and image prediction (`classifyImage()`) must take place in th new thread.

Take a look of a sample code below on image prediction using multithreading:
```python
from imageai.Prediction import ImageClassification
import os
import threading

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsResNet()
prediction.setModelPath( os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))

picturesfolder = os.environ["USERPROFILE"] + "\\Pictures\\"
allfiles = os.listdir(picturesfolder)

class PredictionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        prediction.loadModel()
        for eachPicture in allfiles:
            if eachPicture.endswith(".png") or eachPicture.endswith(".jpg"):
                predictions, percentage_probabilities = prediction.predictImage(picturesfolder + eachPicture, result_count=1)
                for prediction, percentage_probability in zip(predictions, probabilities):
                    print(prediction , " : " , percentage_probability)

predictionThread = PredictionThread ()
predictionThread.start()

```


### Documentation

We have provided full documentation for all **ImageAI** classes and functions in 3 major languages. Find links below:**

* Documentation - **English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)**
* Documentation - **Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)**
* Documentation - **French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)**

