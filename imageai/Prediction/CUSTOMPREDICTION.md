# ImageAI : Custom Image Prediction
A **DeepQuest AI** project <a href="https://deepquestai.com" >https://deepquestai.com </a></p>

---

ImageAI provides 4 different algorithms and model types to perform custom image prediction using your custom models.
You will be able to use your model trained with **ImageAI** and the corresponding model_class JSON file to predict custom objects
that you have trained the model on.

### TABLE OF CONTENTS

- <a href="#customprediction" > :white_square_button: Custom Model Prediction</a>
- <a href="#custompredictionfullmodel" > :white_square_button: Custom Model Prediction with Full Model (NEW)</a>
- <a href="#custompredictionmultiple" > :white_square_button: Custom Prediction with multiple models (NEW)</a>
- <a href="#converttensorflow" > :white_square_button: Convert custom model to Tensorflow's format (NEW)</a>
- <a href="#convertdeepstack" > :white_square_button: Convert custom model to DeepStack's format (NEW)</a>


### Custom Model Prediction
<div id="customprediction"></div>

In this example, we will be using the model trained for 20 experiments on **IdenProf**, a dataset of uniformed professionals and achieved 65.17% accuracy on the test dataset.
(You can use your own trained model and generated JSON file. This 'class' is provided mainly for the purpose to use your own custom models.)
Download the ResNet model of the model and JSON files in links below:

- [**ResNet**](https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof_061-0.7933.h5) _(Size = 90.4 mb)_
- [**IdenProf model_class.json file**](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0.1/model_class.json)

Great!
Once you have downloaded this model file and the JSON file, start a new python project, and then copy the model file and the JSON file to your project folder where your python files (.py files) will be.
Download the image below, or take any image on your computer that include any of the following professionals(Chef, Doctor, Engineer, Farmer, Fireman, Judge, Mechanic, Pilot, Police and Waiter) and copy it to your python project's folder.
Then create a python file and give it a name; an example is **FirstCustomPrediction.py**.
Then write the code below into the python file:

### FirstCustomPrediction.py

```python
from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "idenprof_061-0.7933.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "4.jpg"), result_count=5)

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
from imageai.Prediction.Custom import CustomImagePrediction
import os
```
The code above imports the **ImageAI** library for custom image prediction and the python **os** class.

```python
execution_path = os.getcwd()
```

The above line obtains the path to the folder that contains your python file (in this example, your FirstCustomPrediction.py).

```python
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet_model_ex-020_acc-0.651714.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=10)
```

In the lines above, we created and instance of the `CustomImagePrediction()`
 class in the first line, then we set the model type of the prediction object to ResNet by caling the `.setModelTypeAsResNet()`
  in the second line, we set the model path of the prediction object to the path of the custom model file (`resnet_model_ex-020_acc-0.651714.h5`) we copied to the python file folder
  in the third line, we set the path to  the model_class.json of the model, we load the model and parse the number of objected that can be predicted in the model.

```python
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "4.jpg"), result_count=5)
```

In the above line, we defined 2 variables to be equal to the function called to predict an image, which is the `.predictImage()` function, into which we parsed the path to our image and also state the number of prediction results we want to have (values from 1 to 10 in this case) parsing `result_count=5`. The `.predictImage()` function will return 2 array objects with the first (**predictions**) being an array of predictions and the second (**percentage_probabilities**) being an array of the corresponding percentage probability for each prediction.

```python
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)
```

The above line obtains each object in the **predictions** array, and also obtains the corresponding percentage probability from the **percentage_probabilities**, and finally prints the result of both to console.

**CustomImagePrediction** class also supports the multiple predictions, input types and prediction speeds that are contained
in the **ImagePrediction** class. Follow this [link](README.md) to see all the details.


### Custom Model Prediction with Full Model
<div id="custompredictionfullmodel"></div>

**ImageAI** now allows you to perform prediction using your custom model without specifying the model type. This means you:
1) use your custom, fully saved model that you trained with **ImageAI**
2) use any **Keras** model that is fully saved (with weights and parameters).

All you need to do is load your model using the `loadFullModel()` function instead of `loadModel()`. See the example code below for performing prediction using a fully saved model.

```python
from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()


predictor = CustomImagePrediction()
predictor.setModelPath(model_path=os.path.join(execution_path, "idenprof_full_resnet_ex-001_acc-0.119792.h5"))
predictor.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor.loadFullModel(num_objects=10)

results, probabilities = predictor.predictImage(image_input=os.path.join(execution_path, "1.jpg"), result_count=5)
print(results)
print(probabilities)
```

### Custom Prediction with multiple models
<div id="custompredictionmultiple"></div>


In previous versions of **ImageAI**, running more than one custom model at once wasn't supported.
Now you can run multiple custom models, as many as your computer memory can accommodate.
See the example code below for running multiple custom prediction models.

```python
from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

predictor = CustomImagePrediction()
predictor.setModelPath(model_path=os.path.join(execution_path, "idenprof_resnet.h5"))
predictor.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor.setModelTypeAsResNet()
predictor.loadModel(num_objects=10)

predictor2 = CustomImagePrediction()
predictor2.setModelPath(model_path=os.path.join(execution_path, "idenprof_full_resnet_ex-001_acc-0.119792.h5"))
predictor2.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor2.loadFullModel(num_objects=10)

predictor3 = CustomImagePrediction()
predictor3.setModelPath(model_path=os.path.join(execution_path, "idenprof_inception_0.719500.h5"))
predictor3.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor3.setModelTypeAsInceptionV3()
predictor3.loadModel(num_objects=10)

results, probabilities = predictor.predictImage(image_input=os.path.join(execution_path, "9.jpg"), result_count=5)
print(results)
print(probabilities)

results2, probabilities2 = predictor2.predictImage(image_input=os.path.join(execution_path, "9.jpg"),
                                                    result_count=5)
print(results2)
print(probabilities2)

results3, probabilities3 = predictor3.predictImage(image_input=os.path.join(execution_path, "9.jpg"),
                                                       result_count=5)
print(results3)
print(probabilities3)
print("-------------------------------")
```

### Convert custom model to Tensorflow's format
<div id="converttensorflow"></div>

Using the same `CustomImagePrediction` class you use for custom predictions, you can can now convert your Keras (**.h5**) models into Tensorflow's (**.pb**) model format.
All you need to do is to call the function `save_model_to_tensorflow()` and parse in the necessary parameters as seen in the example code below.

```python
from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()


predictor = CustomImagePrediction()
predictor.setModelPath(model_path=os.path.join(execution_path, "idenprof_full_resnet_ex-001_acc-0.119792.h5"))
predictor.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor.loadFullModel(num_objects=10)

results, probabilities = predictor.predictImage(image_input=os.path.join(execution_path, "1.jpg"), result_count=5)
print(results)
print(probabilities)
```

### Convert custom model to DeepStack's format
<div id="convertdeepstack"></div>

With the `CustomImagePrediction` class you use for custom predictions, you can can now convert your Keras (**.h5**) models into a format deployable on [DeepStack AI Server](https://python.deepstack.cc).
All you need to do is to call the function `save_model_for_deepstack()` and parse in the necessary parameters as seen in the example code below.
It will generate your model and a configuration JSON file.

```python
from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

predictor = CustomImagePrediction()
predictor.setModelTypeAsResNet()
predictor.setModelPath(model_path=os.path.join(execution_path, "idenprof_resnet.h5"))
predictor.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor.loadModel(num_objects=10)
predictor.save_model_for_deepstack(new_model_folder= os.path.join(execution_path, "deepstack_model"), new_model_name="idenprof_resnet_deepstack.h5")
```

### Documentation

We have provided full documentation for all **ImageAI** classes and functions in 3 major languages. Find links below:**

* Documentation - **English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)**
* Documentation - **Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)**
* Documentation - **French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)**

