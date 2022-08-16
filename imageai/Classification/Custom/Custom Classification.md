# ImageAI : Custom Image Classification
A **DeepQuest AI** project <a href="https://deepquestai.com" >https://deepquestai.com </a></p>

---

ImageAI provides 4 different algorithms and model types to perform custom image prediction using your custom models.
You will be able to use your model trained with **ImageAI** and the corresponding model_class JSON file to predict custom objects
that you have trained the model on.

### TABLE OF CONTENTS

- <a href="#customchanges">Changes in ImageAI V2.2.0</a>
- <a href="#customprediction">Custom Model Prediction</a>
- <a href="#custompredictionfullmodel">Custom Model Prediction with Full Model (NEW)</a>
- <a href="#custompredictionmultiple">Custom Prediction with multiple models (NEW)</a>
- <a href="#converttensorflow">Convert custom model to Tensorflow's format (NEW)</a>
- <a href="#convertdeepstack">Convert custom model to DeepStack's format (NEW)</a>

### Custom Model Prediction
<div id="customprediction"></div>

In this example, we will be using the model trained for 20 experiments on **IdenProf**, a dataset of uniformed professionals and achieved 72.0% accuracy on the test dataset.
(You can use your own trained model and generated JSON file. This 'class' is provided mainly for the purpose to use your own custom models.)
Download the ResNet model of the model and JSON files in links below:

- [**ResNet50**](../../../examples/idenprof/model_resnet50_ex-017_acc-0.787_vacc0.747.h5) _(Size = 90.2 mb)_
- [**IdenProf model_class.json file**](../../../examples/idenprof/idenprof.json)

Great!
Once you have downloaded this model file and the JSON file, start a new python project, and then copy the model file and the JSON file to your project folder where your python files (.py files) will be.
Download the image below, or take any image on your computer that include any of the following professionals(Chef, Doctor, Engineer, Farmer, Fireman, Judge, Mechanic, Pilot, Police, or Waiter) and copy it to your python project's folder.
Then create a python file and give it a name; an example is **FirstCustomPrediction.py**.
Then write the code below into the python file:

### FirstCustomPrediction.py

```python
from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()

prediction = CustomImageClassification()
prediction.set_model_type('ResNet50')
prediction.set_model_path(os.path.join(execution_path, "idenprof_resnet50_ex-019_acc-0.773_vacc0.743"))
prediction.set_json_path(os.path.join(execution_path, "idenprof.json"))
prediction.load_trained_model()

predictions, probabilities = prediction.classify_image(os.path.join(execution_path, "4.jpg"), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)
```

**Sample Result:**

![Sample Result](../../../data-images/4.jpg)
```
mechanic : 18.120427429676056
waiter : 9.952791035175323
chef : 9.288407862186432
pilot : 9.214989095926285
doctor : 9.11235511302948
```

You will notice that the probabilities of each prediction are fairly low, this is due to the number of training cycles being only 20. We call these training cycles epochs and when training a new model from scratch 50 epochs is the minimum recommended with 100 epochs being a good place to start and 200 epochs being what you should strive for if you have the computing resources and time.

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
preiction.set_model_type('ResNet50')
prediction.set_model_path(os.path.join(execution_path, "idenprof_resnet50_ex-019_acc-0.773_vacc0.743"))
prediction.set_json_path(os.path.join(execution_path, "idenprof.json"))
prediction.load_full_model()
```

In the lines above, we created and instance of the `CustomImageClassification()` class in the first line, then we set the model type of the prediction object to ResNet by caling the `.set_model_type('ResNet50')` (if `.set_model_type() was not called it would default to ResNet50`) in the second line, we set the model path of the prediction object to the path of the custom model file (`idenprof_resnet50_ex-019_acc-0.773_vacc0.743`) we copied to the python file folder in the third line, we set the path to  the model_class.json of the model, and then we load the model.

```python
predictions, probabilities = prediction.classify_image(os.path.join(execution_path, "4.jpg"), result_count=5)
```

In the above line, we defined 2 variables to be equal to the function called to predict an image, which is the `.classify_image()` function, into which we parsed the path to our image and also state the number of prediction results we want to have (values from 1 to 10 in this case) parsing `result_count=5`. The `.classify_image()` function will return 2 array objects with the first (**predictions**) being an array of predictions and the second (**probabilities**) being an array of the corresponding percentage probability for each prediction.

```python
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)
```

The above line obtains each object in the **predictions** array, and also obtains the corresponding percentage probability from the **probabilities**, and finally prints the result of both to console.

**CustomImageClassification** class also supports the multiple predictions, input types and prediction speeds that are contained
in the **ImageClassification** class. Follow this [link](README.md) to see all the details.



### Custom Prediction with multiple models
<div id="custompredictionmultiple"></div>


In previous versions of **ImageAI**, running more than one custom model at once wasn't supported.
Now you can run multiple custom models, as many as your computer memory can accommodate.
See the example code below for running multiple custom prediction models.

```python
from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()

predictor = CustomImageClassification()
predictor.set_model_path(model_path=os.path.join(execution_path, "idenprof_resnet.h5"))
predictor.set_json_path(model_json=os.path.join(execution_path, "idenprof.json"))
predictor.set_model_type('ResNet50')
predictor.load_model()

predictor2 = CustomImageClassification()
predictor2.set_model_path(model_path=os.path.join(execution_path, "idenprof_inception_0.719500.h5"))
predictor2.set_json_path(model_json=os.path.join(execution_path, "idenprof.json"))
predictor2.setModelTypeAsInceptionV3()
predictor2.load_model(num_objects=10)

results, probabilities = predictor.classify_image(image_input=os.path.join(execution_path, "9.jpg"), result_count=5)
print(results)
print(probabilities)


results2, probabilities2 = predictor2.classify_image(image_input=os.path.join(execution_path, "9.jpg"),
                                                       result_count=5)
print(results2)
print(probabilities2)
print("-------------------------------")
```

<!-- ### Documentation

We have provided full documentation for all **ImageAI** classes and functions in 3 major languages. Find links below:**

* Documentation - **English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)**
* Documentation - **Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)**
* Documentation - **French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)** -->

