# ImageAI : Custom Image Classification

ImageAI provides 4 different algorithms and model types to perform custom image prediction using your custom models.
You will be able to use your model trained with **ImageAI** and the corresponding model_class JSON file to predict custom objects
that you have trained the model on.

### TABLE OF CONTENTS

- <a href="#customprediction" > :white_square_button: Custom Model Prediction</a>
- <a href="#custompredictionfullmodel" > :white_square_button: Custom Model Prediction with Full Model (NEW)</a>

### Custom Model Prediction
<div id="customprediction"></div>

In this example, we will be using the model trained for 20 experiments on **IdenProf**, a dataset of uniformed professionals and achieved 65.17% accuracy on the test dataset.
(You can use your own trained model and generated JSON file. This 'class' is provided mainly for the purpose to use your own custom models.)
Download the ResNet model of the model and JSON files in links below:

- [**ResNet50**](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/resnet50-idenprof-test_acc_0.78200_epoch-91.pt) _(Size = 90.4 mb)_
- [**idenprof_model_class.json file**](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/idenprof_model_classes.json)

Great!
Once you have downloaded this model file and the JSON file, start a new python project, and then copy the model file and the JSON file to your project folder where your python files (.py files) will be.
Download the image below, or take any image on your computer that include any of the following professionals(Chef, Doctor, Engineer, Farmer, Fireman, Judge, Mechanic, Pilot, Police and Waiter) and copy it to your python project's folder.
Then create a python file and give it a name; an example is **FirstCustomPrediction.py**.
Then write the code below into the python file:

### FirstCustomPrediction.py

```python
from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()

prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50-idenprof-test_acc_0.78200_epoch-91.pt"))
prediction.setJsonPath(os.path.join(execution_path, "idenprof_model_class.json"))
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "4.jpg"), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)
```

**Sample Result:**

![Sample Result](../../data-images/4.jpg)
```
mechanic : 76.82620286941528
chef : 10.106072574853897
waiter : 4.036874696612358
police : 2.6663416996598244
pilot : 2.239348366856575
```

The code above works as follows:
```python
from imageai.Classification.Custom import CustomImageClassification
import os
```
The code above imports the **ImageAI** library for custom image prediction and the python **os** class.

```python
execution_path = os.getcwd()
```

The above line obtains the path to the folder that contains your python file (in this example, your FirstCustomPrediction.py).

```python
prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50-idenprof-test_acc_0.78200_epoch-91.pt"))
prediction.setJsonPath(os.path.join(execution_path, "idenprof_model_class.json"))
prediction.loadModel(num_objects=10)
```

In the lines above, we created and instance of the `CustomImageClassification()`
 class in the first line, then we set the model type of the prediction object to ResNet by caling the `.setModelTypeAsResNet50()`
  in the second line, we set the model path of the prediction object to the path of the custom model file (`resnet50-idenprof-test_acc_0.78200_epoch-91.pt`) we copied to the python file folder
  in the third line, we set the path to the idenprof_model_class.json of the model, we load the model and parse the number of objected that can be predicted in the model.

```python
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "4.jpg"), result_count=5)
```

In the above line, we defined 2 variables to be equal to the function called to predict an image, which is the `.classifyImage()` function, into which we parsed the path to our image and also state the number of prediction results we want to have (values from 1 to 10 in this case) parsing `result_count=5`. The `.classifyImage()` function will return 2 array objects with the first (**predictions**) being an array of predictions and the second (**percentage_probabilities**) being an array of the corresponding percentage probability for each prediction.

```python
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)
```

The above line obtains each object in the **predictions** array, and also obtains the corresponding percentage probability from the **percentage_probabilities**, and finally prints the result of both to console.

**CustomImageClassification** class also supports the multiple predictions, input types and prediction speeds that are contained
in the **ImageClassification** class. Follow this [link](README.md) to see all the details.


### Documentation

We have provided full documentation for all **ImageAI** classes and functions in 3 major languages. Find links below:**

* Documentation - **English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)**
