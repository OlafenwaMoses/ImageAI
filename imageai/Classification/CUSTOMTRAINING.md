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
model_trainer.trainModel(num_experiments=100, batch_size=32)
```

In the code above, we start the training process. The parameters stated in the function are as below:
- **num_experiments** : this is to state the number of times the network will train over all the training images,
 which is also called epochs 
- **batch_size** : This is to state the number of images the network will process at ones. The images
 are processed in batches until they are exhausted per each experiment performed. 


When you start the training, you should see something like this in the console:

```
==================================================
Training with GPU
==================================================
Epoch 1/100
----------
100%|█████████████████████████████████████████████████████████████████████████████████| 282/282 [02:15<00:00,  2.08it/s]
train Loss: 3.8062 Accuracy: 0.1178
100%|███████████████████████████████████████████████████████████████████████████████████| 63/63 [00:26<00:00,  2.36it/s]
test Loss: 2.2829 Accuracy: 0.1215
Epoch 2/100
----------
100%|█████████████████████████████████████████████████████████████████████████████████| 282/282 [01:57<00:00,  2.40it/s]
train Loss: 2.2682 Accuracy: 0.1303
100%|███████████████████████████████████████████████████████████████████████████████████| 63/63 [00:20<00:00,  3.07it/s]
test Loss: 2.2388 Accuracy: 0.1470
```

Let us explain the details shown above: 
1. The line **Epoch 1/100** means the network is training the first experiment of the targeted 100 
2. The line `1/25 [>.............................] - ETA: 52s - loss: 2.3026 - acc: 0.2500` represents the number of batches that has been trained in the present experiment
3. The best model is automatically saved to `<dataset-directory>/models>`
 
 Once you are done training your custom model, you can use the "CustomImageClassification" class to perform image prediction with your model. Simply follow the link below.
[imageai/Classification/CUSTOMCLASSIFICATION.md](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Classification/CUSTOMCLASSIFICATION.md)



### Documentation

We have provided full documentation for all **ImageAI** classes and functions. Find links below:

* Documentation - **English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)**
