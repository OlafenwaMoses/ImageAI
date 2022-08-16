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
- 'continue_from_model' now loads a full model and trains from that
- Changed 'enhance_data' to 'preprocess_layers' in trainModel() to be more fitting to the process happening
- Added 'show_training_graph' to trainModel() which is set to False by default. When set to True a graph plotting accuracy with validation accuracy as well as loss with validation loss will show at the end of training
- Moved preprocessing to before the selected model as this is what is recommended in official tensorflow and keras documentation and should improve accuracy when training
- Added RandomZoom() and RandomRotation() to preprocessing to further eliminate overfitting during training
- Removed ImageDataGenerator() as this is depreciated, proprocessing will take the place of this along with rescaling before each model
- Rescaling has been used instead of each models build in preprocessing function as there were issues with training and prediction when using them
- Removed flow_from_directory() as this has been depreceated and image_dataset_from_directory will take its place (this also has the functionality of automatically splitting a single dataset into training and validation datasets automatically)
- Added caching and prefetching of datasets to speed up the training process
- Setting up the chosen model has been simplified and updated to match current best practices in TensorFlow
- On all models 'include_top' has been set to false because of errors when it was set to True
- Saved weights now also contain the name of the model used during training as well as validation accuracy
- Removed tensorboard callback as it wasn't being used

Custom Image Classification:
- set_json_path() is no longer required as long as the json file for the corresponding model is located in the 'json' folder and is named 'model_class.json' 
- Added get_json() which will check if json path is set, if not set will check for 'json/model_class.json' and raise and error if neither are found

Loading Model:
- Simplified code when setting up the model for prediction

Classify Image:
- Changed all keras.preprocessing.image to keras.utils as keras.preprocessing is deprecated
- Added extra processing of the prediction with tf.nn.softmax() as raw data was unreadable
- Unfortunantly due to updates in Tensorflow models from previous generations of ImageAI are no longer going to be supported as the specific model used during training needs to be used during prediction. If you have a Tensorflow SavedModel or HDF5 full model, it can now be used for prediction regardless of if the model was trained in ImageAI or not. If you do not have the 'save_full_model' set to true on models you'd like to continue using, run a few epochs of transfer learning with that model and the 'save_full_model' set to True and it can be carried over to ImageAI V2.2.0 
(This is still being looked into to see if there is a way to transfer models over)