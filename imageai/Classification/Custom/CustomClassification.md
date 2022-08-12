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


### Changes in ImageAI V2.2.0
<div id="customchanges"></div>

ImageAI hasn't been updated since TensorFlow 2.4.0 and were on 2.9.1 now! It's definitely time for some updates and in ImageAI 2.2.0 there were some big changes that involve making using ImageAI as seamless as possible for new users who just want something that works without much configuration while still allowing configuration for users who want to dive deeper into image classification but still aren't ready to build their own models from the ground up or just want something that's easier to use and configure without having to completely re-write your program (I'm the latter and that's why I decided to rewrite it for myself and then wanted to share with everyone else)

NOTE: The custom image classification has been updated quite a bit with changes mostly happening just to get everything working with current versions of tensorflow. With that said, tests were not re-run with the newest version of tensor flow so some of the specific timings of things most likely will not be accurate anymore (they'll usually be faster) but keep that in mind throughout this readme.

**Now time for the changes!**

ImageAI Custom Classification Version 2.2.0 Changes:

Setting models:
- The four model types included with ImageAI are currently still the same as in v2.1.6 however to streamline the package code, make the end user process more simple, and future proof ImageAI, models will no longer be set with setModelTypeAsResNet50(), etc. Models will now be set using set_model_type() with the model you'd like to use ex: trainer.setModelTypeAs('ResNet50')
- On top of changing how the model type is set, setting a model type is no longer required. In almost every case that I've seen, people tend to use ResNet50 so that is now the default model that will be set if set_model_type() is not called. Note: This is mostly to make the experience easier for beginner users

Model training:
- Removed 'num_objects' from trainModel(), this will get calculate based on how many class folders you have in your dataset directory/directories
- Removed 'transfer_with_full_learning' from trainModel() as it wasn't being used
- 'continue_from_model' now loads a full model which isn't a problem since all models are now saved as full models
- Removed 'save_full_model' because HDF5 format will be depreceated in the future and the full model actually takes up less space in my findings than just the weights. This will also allow for pruning and compression to be added in the future to make files even smaller.
- Changed 'enhance_data' to 'preprocess_layers' in trainModel() to be more fitting to the process happening
- Added 'show_training_graph' to trainModel() which is set to False by default. When set to True a graph plotting accuracy with validation accuracy as well as loss with validation loss will show at the end of training
- Moved preprocessing to before the selected model as this is what is recommended in official tensorflow and keras documentation and should improve accuracy when training
- Added RandomZoom() and RanbdomRotation() to preprocessing to further eliminate overfitting during training
- Removed ImageDataGenerator() as this is depreciated, proprocessing will take the place of this along with rescaling before each model
- Rescaling has been used instead of each models build in preprocessing function as there were issues with training and prediction when using them
- Removed flow_from_directory() as this has been depreceated and image_dataset_from_directory will take its place (this also has the functionality of automatically splitting a single dataset into training and validation datasets automatically)
- Added caching and prefetching of datasets to speed up the training process
- Setting up the chosen model has been simplified and updated to match current best practices in TensorFlow
- On all models 'include_top' has been set to false because of errors when it was set to True
- Load weights has been updated to accept weights from models that might not be exactly the same
- Saved weight now also contain the name of the model used during training as well as validation accuracy along with training accuracy still
- Removed tensorboard callback as it wasn't being used

Custom Image Classification:
- set_json_path() is no longer required as long as the json file for the corresponding model is located in the 'json' folder and is named 'model_class.json' 
- Added get_json() which will check if json path is set, if not set will check for 'json/model_class.json' and raise and error if neither are found

Loading Model:
- Simplified code when setting up the model for prediction

Classify Image:
- Changed all keras.preprocessing.image or keras.utils as keras.preprocessing is deprecated
- Added extra processing of the prediction with tf.nn.softmax() as raw data was unreadable
- Unfortunantly due to updates in Tensorflow models from previous generations of ImageAI are no longer going to be supported as the specific model used during training needs to be used during prediction. If you have a Tensorflow SavedModel, it can now be used for prediction regardless of if the model was trained in ImageAI or not. If you do not have the 'save_full_model' set to true on models you'd like to continue using, run a few epochs of transfer learning with that model and the 'save_full_model' set to True and it can be carried over to ImageAI V2.2.0


### Custom Model Prediction
<div id="customprediction"></div>

In this example, we will be using the model trained for 20 experiments on **IdenProf**, a dataset of uniformed professionals and achieved 65.17% accuracy on the test dataset.
(You can use your own trained model and generated JSON file. This 'class' is provided mainly for the purpose to use your own custom models.)
Download the ResNet model of the model and JSON files in links below:

- [**ResNet50**](https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/idenprof_resnet_ex-056_acc-0.993062.h5) _(Size = 90.4 mb)_
- [**IdenProf model_class.json file**](https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/idenprof.json)

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
prediction.set_model_type('ResNet50')
prediction.set_model_path(os.path.join(execution_path, "idenprof_resnet_ex-056_acc-0.993062.h5"))
prediction.set_json_path(os.path.join(execution_path, "idenprof.json"))
prediction.load_model()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "4.jpg"), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)
```

**Sample Result:**

![Sample Result](../../../data-images/4.jpg)
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

The above line obtains the path to the folder that contains your python file (in this example, your FirstCustomPrediction.py). Do note that this step is not completely necessary as the relative path to your root project folder will be used, however it is good practice to eliminate confusion.

```python
prediction = CustomImageClassification()
preiction.set_model_type('ResNet50')
prediction.set_model_path(os.path.join(execution_path, "idenprof_resnet_ex-056_acc-0.993062.h5"))
prediction.set_json_path(os.path.join(execution_path, "idenprof.json"))
prediction.load_model()
```

In the lines above, we created and instance of the `CustomImageClassification()`
 class in the first line, then we set the model type of the prediction object to ResNet by caling the `.set_model_type('ResNet50')` (if `.set_model_type() was not called it would default to ResNet50`) in the second line, we set the model path of the prediction object to the path of the custom model file (`idenprof_resnet_ex-056_acc-0.993062.h5`) we copied to the python file folder in the third line, we set the path to  the model_class.json of the model, and then we load the model.

```python
predictions, probabilities = prediction.classify_image(os.path.join(execution_path, "4.jpg"), result_count=5)
```

In the above line, we defined 2 variables to be equal to the function called to predict an image, which is the `.classify_image()` function, into which we parsed the path to our image and also state the number of prediction results we want to have (values from 1 to 10 in this case) parsing `result_count=5`. The `.classify_image()` function will return 2 array objects with the first (**predictions**) being an array of predictions and the second (**percentage_probabilities**) being an array of the corresponding percentage probability for each prediction.

```python
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)
```

The above line obtains each object in the **predictions** array, and also obtains the corresponding percentage probability from the **percentage_probabilities**, and finally prints the result of both to console.

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


results2, probabilities2 = predictor3.classify_image(image_input=os.path.join(execution_path, "9.jpg"),
                                                       result_count=5)
print(results2)
print(probabilities2)
print("-------------------------------")
```

### Documentation

We have provided full documentation for all **ImageAI** classes and functions in 3 major languages. Find links below:**

* Documentation - **English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)**
* Documentation - **Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)**
* Documentation - **French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)**

