# ImageAI : Custom Prediction Model Training 

---

**ImageAI** provides the most simple and powerful approach to training custom image prediction models
using state-of-the-art SqueezeNet, ResNet50, InceptionV3 and DenseNet
which you can load into the `imageai.Classification.Custom.CustomImageClassification` class. This allows
 you to train your own model on any set of images that corresponds to any type of objects/persons.
The training process generates a JSON file that maps the objects types in your image dataset
and creates lots of models. You will then pick the model with the highest accuracy and perform custom
image prediction using the model and the JSON file generated.

### TABLE OF CONTENTS
- <a href="#customtraining" > :white_square_button: Custom Model Training Prediction</a> 
- <a href="#savefullmodel" > :white_square_button: Saving Full Custom Model </a> 
- <a href="#idenproftraining" > :white_square_button: Training on the IdenProf Dataset</a> 
- <a href="#continuoustraining" > :white_square_button: Continuous Model Training </a> 
- <a href="#transferlearning" > :white_square_button: Transfer Learning (Training from a pre-trained model)</a>


### Custom Model Training
<div id="customtraining"></div>

Because model training is a compute intensive tasks, we strongly advise you perform this experiment using a computer with a NVIDIA GPU and the GPU version of Tensorflow installed. Performing model training on CPU will my take hours or days. With NVIDIA GPU powered computer system, this will take a few hours.  You can use Google Colab for this experiment as it has an NVIDIA K80 GPU available.

To train a custom prediction model, you need to prepare the images you want to use to train the model.
You will prepare the images as follows:

1. Create a dataset folder with the name you will like your dataset to be called (e.g pets) 
2. In the dataset folder, create a folder by the name **train** 
3. In the dataset folder, create a folder by the name **test** 
4. In the train folder, create a folder for each object you want to the model to predict and give the folder a name that corresponds to the respective object name (e.g dog, cat, squirrel, snake) 
5. In the test folder, create a folder for each object you want to the model to predict and give
 the folder a name that corresponds to the respective object name (e.g dog, cat, squirrel, snake) 
6. In each folder present in the train folder, put the images of each object in its respective folder. This images are the ones to be used to train the model To produce a model that can perform well in practical applications, I recommend you about 500 or more images per object. 1000 images per object is just great 
7. In each folder present in the test folder, put about 100 to 200 images of each object in its respective folder. These images are the ones to be used to test the model as it trains 
8. Once you have done this, the structure of your image dataset folder should look like below:  
    ```
    pets//train//dog//dog-train-images
    pets//train//cat//cat-train-images
    pets//train//squirrel//squirrel-train-images
    pets//train//snake//snake-train-images 
    pets//test//dog//dog-test-images
    pets//test//cat//cat-test-images
    pets//test//squirrel//squirrel-test-images
    pets//test//snake//snake-test-images
    ```
9. Then your training code goes as follows:  
    ```python
    from imageai.Classification.Custom import ClassificationModelTrainer
    model_trainer = ClassificationModelTrainer()
    model_trainer.setModelTypeAsResNet50()
    model_trainer.setDataDirectory("pets")
    model_trainer.trainModel(num_objects=4, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)
    ```

 Yes! Just 5 lines of code and you can train any of the available 4 state-of-the-art Deep Learning algorithms on your custom dataset.
Now lets take a look at how the code above works.

```python
from imageai.Classification.Custom import ClassificationModelTrainer
model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory("pets")
```

In the first line, we import the **ImageAI** model training class, then we define the model trainer in the second line,
 we set the network type in the third line and set the path to the image dataset we want to train the network on.

```python
model_trainer.trainModel(num_objects=4, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)
```

In the code above, we start the training process. The parameters stated in the function are as below:
- **num_objects** : this is to state the number of object types in the image dataset 
- **num_experiments** : this is to state the number of times the network will train over all the training images,
 which is also called epochs 
- **enhance_data (optional)** : This is used to state if we want the network to produce modified copies of the training
images for better performance. 
- **batch_size** : This is to state the number of images the network will process at ones. The images
 are processed in batches until they are exhausted per each experiment performed. 
- **show_network_summary** : This is to state if the network should show the structure of the training
 network in the console.
 

When you start the training, you should see something like this in the console:
```
Total params: 23,608,202
Trainable params: 23,555,082
Non-trainable params: 53,120
____________________________________________________________________________________________________
Using Enhanced Data Generation
Found 4000 images belonging to 4 classes.
Found 800 images belonging to 4 classes.
JSON Mapping for the model classes saved to  C:\Users\User\PycharmProjects\ImageAITest\pets\json\model_class.json
Number of experiments (Epochs) :  100
```

When the training progress progresses, you will see results as follows in the console: 
```
Epoch 1/100
 1/25 [>.............................] - ETA: 52s - loss: 2.3026 - acc: 0.2500
 2/25 [=>............................] - ETA: 41s - loss: 2.3027 - acc: 0.1250
 3/25 [==>...........................] - ETA: 37s - loss: 2.2961 - acc: 0.1667
 4/25 [===>..........................] - ETA: 36s - loss: 2.2980 - acc: 0.1250
 5/25 [=====>........................] - ETA: 33s - loss: 2.3178 - acc: 0.1000
 6/25 [======>.......................] - ETA: 31s - loss: 2.3214 - acc: 0.0833
 7/25 [=======>......................] - ETA: 30s - loss: 2.3202 - acc: 0.0714
 8/25 [========>.....................] - ETA: 29s - loss: 2.3207 - acc: 0.0625
 9/25 [=========>....................] - ETA: 27s - loss: 2.3191 - acc: 0.0556
10/25 [===========>..................] - ETA: 25s - loss: 2.3167 - acc: 0.0750
11/25 [============>.................] - ETA: 23s - loss: 2.3162 - acc: 0.0682
12/25 [=============>................] - ETA: 21s - loss: 2.3143 - acc: 0.0833
13/25 [==============>...............] - ETA: 20s - loss: 2.3135 - acc: 0.0769
14/25 [===============>..............] - ETA: 18s - loss: 2.3132 - acc: 0.0714
15/25 [=================>............] - ETA: 16s - loss: 2.3128 - acc: 0.0667
16/25 [==================>...........] - ETA: 15s - loss: 2.3121 - acc: 0.0781
17/25 [===================>..........] - ETA: 13s - loss: 2.3116 - acc: 0.0735
18/25 [====================>.........] - ETA: 12s - loss: 2.3114 - acc: 0.0694
19/25 [=====================>........] - ETA: 10s - loss: 2.3112 - acc: 0.0658
20/25 [=======================>......] - ETA: 8s - loss: 2.3109 - acc: 0.0625
21/25 [========================>.....] - ETA: 7s - loss: 2.3107 - acc: 0.0595
22/25 [=========================>....] - ETA: 5s - loss: 2.3104 - acc: 0.0568
23/25 [==========================>...] - ETA: 3s - loss: 2.3101 - acc: 0.0543
24/25 [===========================>..] - ETA: 1s - loss: 2.3097 - acc: 0.0625Epoch 00000: saving model to C:\Users\Moses\Documents\Moses\W7\AI\Custom Datasets\IDENPROF\idenprof-small-test\idenprof\models\model_ex-000_acc-0.100000.h5

25/25 [==============================] - 51s - loss: 2.3095 - acc: 0.0600 - val_loss: 2.3026 - val_acc: 0.1000
```

Let us explain the details shown above: 
1. The line **Epoch 1/100** means the network is training the first experiment of the targeted 100 
2. The line `1/25 [>.............................] - ETA: 52s - loss: 2.3026 - acc: 0.2500` represents the number of batches that has been trained in the present experiment
3. The line  `Epoch 00000: saving model to C:\Users\User\PycharmProjects\ImageAITest\pets\models\model_ex-000_acc-0.100000.h5` refers to the model saved after the present experiment. The **ex_000** represents the experiment at this stage while the **acc_0.100000** and **val_acc: 0.1000** represents the accuracy of the model on the test images after the present experiment (maximum value value of accuracy is 1.0).  This result helps to know the best performed model you can use for custom image prediction.  
 
 Once you are done training your custom model, you can use the "CustomImagePrediction" class to perform image prediction with your model. Simply follow the link below.
[imageai/Classification/CUSTOMCLASSIFICATION.md](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Classification/CUSTOMCLASSIFICATION.md)


### Training on the IdenProf data

A sample from the IdenProf Dataset used to train a Model for predicting professionals.
![](../../data-images/idenprof.jpg)

Below we provide a sample code to train on **IdenProf**, a dataset which contains images of 10 uniformed professionals. The code below will download the dataset and initiate the training:

```python
from io import open
import requests
import shutil
from zipfile import ZipFile
import os
from imageai.Classification.Custom import ClassificationModelTrainer

execution_path = os.getcwd()

TRAIN_ZIP_ONE = os.path.join(execution_path, "idenprof-train1.zip")
TRAIN_ZIP_TWO = os.path.join(execution_path, "idenprof-train2.zip")
TEST_ZIP = os.path.join(execution_path, "idenprof-test.zip")

DATASET_DIR = os.path.join(execution_path, "idenprof")
DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, "train")
DATASET_TEST_DIR = os.path.join(DATASET_DIR, "test")

if(os.path.exists(DATASET_DIR) == False):
    os.mkdir(DATASET_DIR)
if(os.path.exists(DATASET_TRAIN_DIR) == False):
    os.mkdir(DATASET_TRAIN_DIR)
if(os.path.exists(DATASET_TEST_DIR) == False):
    os.mkdir(DATASET_TEST_DIR)

if(len(os.listdir(DATASET_TRAIN_DIR)) < 10):
    if(os.path.exists(TRAIN_ZIP_ONE) == False):
        print("Downloading idenprof-train1.zip")
        data = requests.get("https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-train1.zip", stream = True)
        with open(TRAIN_ZIP_ONE, "wb") as file:
            shutil.copyfileobj(data.raw, file)
        del data
    if (os.path.exists(TRAIN_ZIP_TWO) == False):
        print("Downloading idenprof-train2.zip")
        data = requests.get("https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-train2.zip", stream=True)
        with open(TRAIN_ZIP_TWO, "wb") as file:
            shutil.copyfileobj(data.raw, file)
        del data
    print("Extracting idenprof-train1.zip")
    extract1 = ZipFile(TRAIN_ZIP_ONE)
    extract1.extractall(DATASET_TRAIN_DIR)
    extract1.close()
    print("Extracting idenprof-train2.zip")
    extract2 = ZipFile(TRAIN_ZIP_TWO)
    extract2.extractall(DATASET_TRAIN_DIR)
    extract2.close()

if(len(os.listdir(DATASET_TEST_DIR)) < 10):
    if (os.path.exists(TEST_ZIP) == False):
        print("Downloading idenprof-test.zip")
        data = requests.get("https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-test.zip", stream=True)
        with open(TEST_ZIP, "wb") as file:
            shutil.copyfileobj(data.raw, file)
        del data
    print("Extracting idenprof-test.zip")
    extract = ZipFile(TEST_ZIP)
    extract.extractall(DATASET_TEST_DIR)
    extract.close()


model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory(DATASET_DIR)
model_trainer.trainModel(num_objects=10, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)
```

### Continuous Model Training
<div id="continuoustraining"></div>

**ImageAI** now allows you to continue training your custom model on your previously saved model.
This is useful in cases of incomplete training due compute time limits/large size of dataset or should you intend to further train your model.
Kindly note that **continuous training** is for using a previously saved model to train on the same dataset the model was trained on.
All you need to do is specify the `continue_from_model` parameter to the path of the previously saved model in your `trainModel()` function.
See an example code below.

```python
from imageai.Classification.Custom import ClassificationModelTrainer
import os

trainer = ClassificationModelTrainer()
trainer.setModelTypeAsDenseNet121()
trainer.setDataDirectory("idenprof")
trainer.trainModel(num_objects=10, num_experiments=50, enhance_data=True, batch_size=8, show_network_summary=True, continue_from_model="idenprof_densenet-0.763500.h5")
```

### Transfer Learning (Training from a pre-trained model)
<div id="transferlearning"></div>

From the feedbacks we have received over the past months, we discovered most custom models trained with **ImageAI** were based on datasets with few number of images as they fall short the minimum recommendation of 500 images per each class of objects, for a achieving a viable accuracy. 

To ensure they can still train very accurate custom models using few number of images, **ImageAI** now allows you to train by leveraging **transfer learning** . This means you can take any pre-trained **ResNet50**, **Squeezenet**, **InceptionV3** and **DenseNet121** model trained on larger datasets and use it to kickstart your custom model training.
All you need to do is specify the `transfer_from_model` parameter to the path of the pre-trained model, `initial_num_objects` parameter which corresponds to the number of objects in the previous dataset the pre-trained model was trained on, all in your `trainModel()` function. See an example code below, showing how to perform transfer learning from a ResNet50 model trained on the ImageNet dataset.

```python
from imageai.Classification.Custom import ClassificationModelTrainer
import os

trainer = ClassificationModelTrainer()
trainer.setModelTypeAsResNet50()
trainer.setDataDirectory("idenprof")
trainer.trainModel(num_objects=10, num_experiments=50, enhance_data=True, batch_size=32, show_network_summary=True,transfer_from_model="resnet50_imagenet_tf.2.0.h5", initial_num_objects=1000)
```


### Contact Developer
- **Moses Olafenwa**
    * _Email:_ guymodscientist@gmail.com
    * _Website:_ [https://moses.aicommons.science](https://moses.aicommons.science)
    * _Twitter:_ [@OlafenwaMoses](https://twitter.com/OlafenwaMoses)
    * _Medium:_ [@guymodscientist](https://medium.com/@guymodscientist)
    * _Facebook:_ [moses.olafenwa](https://facebook.com/moses.olafenwa)


### Documentation

We have provided full documentation for all **ImageAI** classes and functions in 3 major languages. Find links below:

* Documentation - **English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)**
* Documentation - **Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)**
* Documentation - **French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)**
