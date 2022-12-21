# ImageAI : Image Prediction
A **DeepQuest AI** project [https://deepquestai.com](https://deepquestai.com)

---

### TABLE OF CONTENTS
- <a href="#firstprediction" > :white_square_button: First Prediction</a>
- <a href="#documentation" > :white_square_button: Documentation</a>

ImageAI provides 4 different algorithms and model types to perform image prediction.
To perform image prediction on any picture, take the following simple steps.  The 4 algorithms provided for
 image prediction include **MobileNetV2**, **ResNet50**, **InceptionV3** and **DenseNet121**. Each of these
  algorithms have individual model files which you must use depending on the choice of your algorithm. To download the
   model file for your choice of algorithm, click on any of the links below:
   
- **[MobileNetV2](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/mobilenet_v2-b0353104.pth)** _(Size = 4.82 mb, fastest prediction time and moderate accuracy)_
- **[ResNet50](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/resnet50-19c8e357.pth)** by Microsoft Research _(Size = 98 mb, fast prediction time and high accuracy)_
 - **[InceptionV3](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/inception_v3_google-1a9a5a14.pth)** by Google Brain team _(Size = 91.6 mb, slow prediction time and higher accuracy)_
 - **[DenseNet121](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/densenet121-a639ec97.pth)** by Facebook AI Research _(Size = 31.6 mb, slower prediction time and highest accuracy)_

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
prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth"))
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
prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth"))
```
In the lines above, we created and instance of the `ImagePrediction()` class in the first line, then we set the model type of the prediction object to ResNet by caling the `.setModelTypeAsResNet50()` in the second line and then we set the model path of the prediction object to the path of the model file (`resnet50-19c8e357.pth`) we copied to the python file folder in the third line.

```python
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "1.jpg"), result_count=5 )
```

In the above line, we defined 2 variables to be equal to the function called to predict an image, which is the `.classifyImage()` function, into which we parsed the path to our image and also state the number of prediction results we want to have (values from 1 to 1000) parsing `result_count=5`. The `.classifyImage()` function will return 2 array objects with the first (**predictions**) being an array of predictions and the second (**percentage_probabilities**) being an array of the corresponding percentage probability for each prediction.

```python
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : " , eachProbability)
```
The above line obtains each object in the **predictions** array, and also obtains the corresponding percentage probability from the **percentage_probabilities**, and finally prints the result of both to console.




### Documentation

We have provided full documentation for all **ImageAI** classes and functions. Find links below:**

* Documentation - **English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)**

