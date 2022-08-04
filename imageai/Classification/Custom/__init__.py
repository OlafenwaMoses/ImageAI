import tensorflow as tf
from PIL import Image
import time
import numpy as np
import os
import warnings
from matplotlib.cbook import deprecated
import json

class ClassificationModelTrainer:

    """
        This is the Classification Model training class, that allows you to define a deep learning network
        from the 4 available networks types supported by ImageAI which are MobileNetv2, ResNet50,
        InceptionV3 and DenseNet121.
    """

    def __init__(self):
        self.__modelType = ""
        self.__use_pretrained_model = False
        self.__data_dir = ""
        self.__single_dataset_dir = False
        self.__validation_split = 0.2
        self.__dataset_dir = ''
        self.__train_dir = ""
        self.__test_dir = ""
        self.__logs_dir = ""
        self.__num_epochs = 10
        self.__trained_model_dir = ""
        self.__model_class_dir = ""
        self.__initial_learning_rate = 1e-3
        self.__model_collection = []


    def setModelType(self, modelType):

        # Take the model type input and make it lowercase to remove any capitalizaton errors then set model type variable
        match modelType.lower():
            case 'mobilenetv2':
                self.__modelType = 'MobileNetV2'
            case 'resnet50':
                self.__modelType = 'ResNet50'
            case 'densenet121':
                self.__modelType = 'DenseNet121'
            case 'inceptionv3':
                self.__modelType = 'InceptionV3'


    def setDataDirectory(self, data_directory="", single_dataset_directory=False, dataset_directory='data', validation_split=0.2, train_subdirectory="train", test_subdirectory="test",
                         models_subdirectory="models", json_subdirectory="json"):
        """
        'setDataDirectory()'

        - data_directory , is required to set the path to which the data/dataset to be used for
                 training is kept. The directory can have any name, but it must have 'train' and 'test'
                 sub-directory. In the 'train' and 'test' sub-directories, there must be sub-directories
                 with each having it's name corresponds to the name/label of the object whose images are
                to be kept. The structure of the 'test' and 'train' folder must be as follows:

                >> train >> class1 >> class1_train_images
                         >> class2 >> class2_train_images
                         >> class3 >> class3_train_images
                         >> class4 >> class4_train_images
                         >> class5 >> class5_train_images

                >> test >> class1 >> class1_test_images
                        >> class2 >> class2_test_images
                        >> class3 >> class3_test_images
                        >> class4 >> class4_test_images
                        >> class5 >> class5_test_images

        - train_subdirectory (optional), subdirectory within 'data_directory' where the training set is. Defaults to 'train'.
        - test_subdirectory (optional), subdirectory within 'data_directory' where the testing set is. Defaults to 'test'.
        - models_subdirectory (optional), subdirectory within 'data_directory' where the output models will be saved. Defaults to 'models'.
        - json_subdirectory (optional), subdirectory within 'data_directory' where the model classes json file will be saved. Defaults to 'json'.

        :param data_directory:
        :param train_subdirectory:
        :param test_subdirectory:
        :param models_subdirectory:
        :param json_subdirectory:
        :return:
        """

        self.__data_dir = data_directory

        self.__single_dataset_dir = single_dataset_directory
        self.__dataset_dir = os.path.join(self.__data_dir, dataset_directory)
        self.__validation_split = validation_split

        self.__train_dir = os.path.join(self.__data_dir, train_subdirectory)
        self.__test_dir = os.path.join(self.__data_dir, test_subdirectory)
        self.__trained_model_dir = os.path.join(self.__data_dir, models_subdirectory)
        self.__model_class_dir = os.path.join(self.__data_dir, json_subdirectory)
        self.__logs_dir = os.path.join(self.__data_dir, "logs")

    def lr_schedule(self, epoch):

        # Learning Rate Schedule
        lr = self.__initial_learning_rate
        total_epochs = self.__num_epochs

        check_1 = int(total_epochs * 0.9)
        check_2 = int(total_epochs * 0.8)
        check_3 = int(total_epochs * 0.6)
        check_4 = int(total_epochs * 0.4)

        if epoch > check_1:
            lr *= 1e-4
        elif epoch > check_2:
            lr *= 1e-3
        elif epoch > check_3:
            lr *= 1e-2
        elif epoch > check_4:
            lr *= 1e-1


        return lr




    def trainModel(self, num_experiments=200, enhance_data=False, batch_size = 32, initial_learning_rate=1e-3, show_network_summary=False, training_image_size = 224, continue_from_model=None, transfer_from_model=None, transfer_with_full_training=True, save_full_model = False):

        """
        'trainModel()' function starts the model actual training. It accepts the following values:
        - num_experiments , also known as epochs, it is the number of times the network will train on all the training dataset
        - enhance_data (optional) , this is used to modify the dataset and create more instance of the training set to enhance the training result
        - batch_size (optional) , due to memory constraints, the network trains on a batch at once, until all the training set is exhausted. The value is set to 32 by default, but can be increased or decreased depending on the meormory of the compute used for training. The batch_size is conventionally set to 16, 32, 64, 128.
        - initial_learning_rate(optional) , this value is used to adjust the weights generated in the network. You rae advised to keep this value as it is if you don't have deep understanding of this concept.
        - show_network_summary(optional) , this value is used to show the structure of the network should you desire to see it. It is set to False by default
        - training_image_size(optional) , this value is used to define the image size on which the model will be trained. The value is 224 by default and is kept at a minimum of 100.
        - continue_from_model (optional) , this is used to set the path to a model file trained on the same dataset. It is primarily for continuos training from a previously saved model.
        - transfer_from_model (optional) , this is used to set the path to a model file trained on another dataset. It is primarily used to perform tramsfer learning.
        - transfer_with_full_training (optional) , this is used to set the pre-trained model to be re-trained across all the layers or only at the top layers.
        - save_full_model ( optional ), this is used to save the trained models with their network types. Any model saved by this specification can be loaded without specifying the network type.


        :param num_experiments:
        :param enhance_data:
        :param batch_size:
        :param initial_learning_rate:
        :param show_network_summary:
        :param training_image_size:
        :param continue_from_model:
        :param transfer_from_model:
        :param save_full_model:
        :return:
        """

        self.__num_epochs = num_experiments
        self.__initial_learning_rate = initial_learning_rate
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)


        if(training_image_size < 100):
            warnings.warn("The specified training_image_size {} is less than 100. Hence the training_image_size will default to 100.".format(training_image_size))
            training_image_size = 100


        input_shape = (training_image_size,  training_image_size, 3)

        
        # Preprocess layers for image scaling
        # If enhance data is set to True add extra preprocessing for better accuracy
        def preprocess_inputs(inputs):
            if enhance_data:
                x = tf.keras.layers.RandomFlip("horizontal", input_shape=input_shape)(inputs)
                x = tf.keras.layers.RandomRotation(0.1)(x)
                x = tf.keras.layers.RandomZoom(0.1)(x)
                x = tf.keras.layers.RandomHeight(0.1)(x)
                x = tf.keras.layers.RandomWidth(0.1)(x)

            return x
        

        # Set up training and validation datasets with caching and prefetching for improved training performance
        AUTOTUNE = tf.data.AUTOTUNE

        if self.__single_dataset_dir == True:
            print('Splitting the dataset into training and validation subsets automatically')

        train_ds = tf.keras.utils.image_dataset_from_directory(
          self.__train_dir if self.__single_dataset_dir == False else self.__dataset_dir,
          validation_split = None if self.__single_dataset_dir == False else self.__validation_split,
          subset = None if self.__single_dataset_dir == False else 'training',
          seed=123,
          image_size=(training_image_size, training_image_size),
          batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
          self.__test_dir if self.__single_dataset_dir == False else self.__dataset_dir,
          validation_split = None if self.__single_dataset_dir == False else self.__validation_split,
          subset = None if self.__single_dataset_dir == False else 'validation',
          seed=123,
          image_size=(training_image_size, training_image_size),
          batch_size=batch_size)

        class_names = train_ds.class_names
        num_classes = len(class_names)

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        # Set up inputs and preprocess them
        inputs = tf.keras.Input(shape=input_shape)
        outputs = inputs
        if enhance_data == True:
            print("Using Enhanced Data Generation")
            outputs = preprocess_inputs(inputs)


        # Check for model type and build a model based on it
        if self.__modelType == '':
            print('No model type specified, defaulting to ResNet50')
            self.__modelType = 'ResNet50'
        match self.__modelType:
            case 'MobileNetV2':
                outputs = tf.keras.layers.Rescaling(1./127.5, input_shape=input_shape)(outputs)
                outputs = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, weights= None, classes=num_classes,
                                                            include_top=False, pooling='avg')(outputs)
            case 'ResNet50':
                outputs = tf.keras.layers.Rescaling(1./255, input_shape=input_shape)(outputs)
                outputs = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape, weights= None, classes=num_classes,
                                                            include_top=False, pooling='avg')(outputs)
            case 'DenseNet121':
                outputs = tf.keras.layers.Rescaling(1./255, input_shape=input_shape)(outputs)
                outputs = tf.keras.applications.densenet.DenseNet121(input_shape=input_shape, weights= None, classes=num_classes,
                                                            include_top=False, pooling='avg')(outputs)
            case 'InceptionV3':
                outputs = tf.keras.layers.Rescaling(1./127.5, input_shape=input_shape)(outputs)
                outputs = tf.keras.applications.inception_v3.InceptionV3(input_shape=input_shape, weights= None, classes=num_classes,
                                                            include_top=False, pooling='avg')(outputs)


        # Build final output and create model
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=True)(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)


        # Compile the model to get ready for training
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])
        

        # If continuing or transfering a model set load those weights
        if continue_from_model != None or transfer_from_model != None:
            model.load_weights(continue_from_model if continue_from_model != None else transfer_from_model, skip_mismatch=True, by_name=True)

        
        # Print model summary if set and print if using a previous model
        if (show_network_summary == True):
            if continue_from_model != None or transfer_from_model != None:
                print('Training using weights from a previous model')
            model.summary()


        # Set templates for model name, model path, log name, and log path
        model_name = 'model_ex-{epoch:03d}_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}.h5'
        model_path = os.path.join(self.__trained_model_dir, model_name)
        log_name = '{}_lr-{}_{}'.format(self.__modelType, initial_learning_rate, time.strftime("%Y-%m-%d-%H-%M-%S"))
        logs_path = os.path.join(self.__logs_dir, log_name)

        
        # Check if data directories exist and if they don't create them
        directories = [self.__trained_model_dir, self.__model_class_dir, self.__logs_dir, os.path.join(self.__logs_dir, log_name)]
        for dirs in directories:
            if not os.path.isdir(dirs):
                os.makedirs(dirs)


        # Check for saving the full model
        save_weights_condition = True
        if(save_full_model == True):
            save_weights_condition = False


        # Create a checkpoint callback to save models as they're completed
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                     monitor='accuracy',
                                     verbose=1,
                                     save_weights_only=save_weights_condition,
                                     save_best_only=True)
        

        # Create a model class json file
        class_json = {}
        for i, class_name in enumerate(class_names):
            class_json[str(i)] = class_name 
        with open(os.path.join(self.__model_class_dir,'model_class.json'), 'w+') as json_file:
            json.dump(class_json, json_file, indent=4, separators=(',', ' : '),
                      ensure_ascii=True)
        print("JSON Mapping for the model classes saved to ", os.path.join(self.__model_class_dir, "model_class.json"))


        # Print number of experiments that will be run
        print("Number of experiments (Epochs) : ", self.__num_epochs)

        
        # Train the model
        model.fit(train_ds, epochs=self.__num_epochs,
                            validation_data=val_ds,
                            callbacks=[checkpoint, lr_scheduler])
                


    def setModelTypeAsSqueezeNet(self):
        raise ValueError("ImageAI no longer support SqueezeNet. You can use MobileNetV2 instead by downloading the MobileNetV2 model and call the function 'setModelTypeAsMobileNetV2'")

    @deprecated(since="2.1.6", message="'.predictImage()' has been deprecated! Please use 'classifyImage()' instead.")
    def predictImage(self, image_input, result_count=3, input_type="file"):
        return self.classifyImage(image_input, result_count, input_type)

    @deprecated(since="2.1.6", message="'.setModelTypeAsResNet()' has been deprecated! Please use 'setModelType('ResNet50')' instead.")
    def setModelTypeAsResNet(self):
        return self.setModelTypeAsResNet50()

    @deprecated(since="2.1.6", message="'.setModelTypeAsDenseNet()' has been deprecated! Please use 'setModelType('DenseNet121')' instead.")
    def setModelTypeAsDenseNet(self):
        return self.setModelType('DenseNet121')

    @deprecated(since="2.2.0", message="'.setModelTypeAsMobileNetV2()' has been deprecated! Please use 'setModelType('MobileNetV2')' instead.")
    def setModelTypeAsMobileNetV2(self):
        return self.setModelType('MobileNetV2')

    @deprecated(since="2.2.0", message="'.setModelTypeAsResNet50()' has been deprecated! Please use 'setModelType('ResNet50')' instead.")
    def setModelTypeAsResNet50(self):
        return self.setModelType('ResNet50')

    @deprecated(since="2.2.0", message="'.setModelTypeAsDenseNet121()' has been deprecated! Please use 'setModelType('DenseNet121')' instead.")
    def setModelTypeAsDenseNet121(self):
        return self.setModelType('DenseNet121')

    @deprecated(since="2.2.0", message="'.setModelTypeAsInceptionV3()' has been deprecated! Please use 'setModelType('InceptionV3')' instead.")
    def setModelTypeAsInceptionV3(self):
        return self.setModelType('InceptionV3')





class CustomImageClassification:

    """
    This is the image classification class for custom models trained with the 'ClassificationModelTrainer' class. It provides support for 4 different models which are:
    ResNet50, MobileNetV2, DenseNet121 and Inception V3. After instantiating this class, you can set it's properties and
    make image classification using it's pre-defined functions.

    The following functions are required to be called before a classification can be made
    * setModelPath() , path to your custom model
    * setJsonPath , , path to your custom model's corresponding JSON file
    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsMobileNetV2(), setModelTypeAsResNet50(), setModelTypeAsDenseNet121, setModelTypeAsInceptionV3]
    * loadModel() [This must be called once only before making a classification]

    Once the above functions have been called, you can call the classifyImage() function of the classification instance
    object at anytime to predict an image.
    """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.jsonPath = ""
        self.numObjects = 10
        self.__model_classes = dict()
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_size = 224
    


    def setModelPath(self, model_path):

        """
        'setModelPath()' function is required and is used to set the file path to the model adopted from the list of the
        available 4 model types. The model path must correspond to the model type set for the classification instance object.

        :param model_path:
        :return:
        """

        self.modelPath = model_path



    def setJsonPath(self, model_json):

        """
        'setJsonPath()' function is not required unless model json is not named 'model_class.json' in the 'json' directory of project root folder.

        :param model_path:
        :return:
        """

        self.jsonPath = model_json


    def setModelType(self, modelType):

        # Take the model type input and make it lowercase to remove any capitalizaton errors then set model type variable
        match modelType.lower():
            case 'mobilenetv2':
                self.__modelType = 'MobileNetV2'
            case 'resnet50':
                self.__modelType = 'ResNet50'
            case 'densenet121':
                self.__modelType = 'DenseNet121'
            case 'inceptionv3':
                self.__modelType = 'InceptionV3'


    def getJson(self):

        # If json path is specified open it, otherwise check the default location
        # If json path is not specified nor in the default location raise ValueError
        if self.jsonPath != '':
            self.__model_classes = json.load(open(self.jsonPath))
        else:
            try:
                self.__model_classes = json.load(open('json/model_class.json'))
            except:
                raise ValueError('There was an error when loading the model class json file. Make sure the file location is \'json\\model_class.json\' or set the json path with setJsonPath()')


    def loadModel(self, classification_speed="normal"):

        """
        'loadModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. This function receives an optional value which is "classification_speed".
        The value is used to reduce the time it takes to classify an image, down to about 50% of the normal time,
        with just slight changes or drop in classification accuracy, depending on the nature of the image.
        * classification_speed (optional); Acceptable values are "normal", "fast", "faster" and "fastest"

        :param classification_speed :
        :return:
        """

        # Call getJson to load all model classes
        self.getJson()


        # Adjust the size that the input image will be rescaled to
        # Smaller numbers mean smaller picture which means quicker analysis and prediction
        # The trade-off to faster speeds is prediction accuracy
        match classification_speed:
            case 'normal':
                self.__input_image_size = 224
            case 'fast':
                self.__input_image_size = 160
            case 'faster':
                self.__input_image_size = 120
            case 'fastest':
                self.__input_image_size = 100


        # Create model inputs then rescale inputs to image size which is used by all models
        inputs = tf.keras.Input(shape=(self.__input_image_size, self.__input_image_size, 3))
        rescaled = tf.keras.layers.Rescaling(1./255, input_shape=(self.__input_image_size, self.__input_image_size, 3))(inputs)


        # Only load model if a model hasn't already been loaded
        if (self.__modelLoaded == False):

            # Get number of objects from model class list length
            num_classes = len(self.__model_classes)
            try:

                # Set outputs based on model type
                if self.__modelType == '':
                    print('No model type specified, defaulting to ResNet50')
                    self.__modelType = 'ResNet50'
                match self.__modelType:
                    case 'MobileNetV2':
                        outputs = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes=num_classes, include_top=False, pooling='avg')(rescaled)
                    case 'ResNet50':
                        outputs = tf.keras.applications.resnet50.ResNet50(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes=num_classes, include_top=False, pooling='avg')(rescaled)
                    case 'DenseNet121':
                        outputs = tf.keras.applications.densenet.DenseNet121(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes=num_classes, include_top=False, pooling='avg')(rescaled)
                    case 'InceptionV3':
                        outputs = tf.keras.applications.inception_v3.InceptionV3(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes=num_classes, include_top=False, pooling='avg')(rescaled)
                    case other:
                        raise ValueError("You must set a valid model type before loading the model.")

                # Create, compile, then load weights into the model
                outputs = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=True)(outputs)
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
                model.load_weights(self.modelPath)
                self.__model_collection.append(model)
                self.__modelLoaded = True
                
            except:
                raise ValueError("An error occured. Ensure your model file is a {} Model and is located in the path {}".format(self.__modelType, self.modelPath))


    def loadFullModel(self, classification_speed="normal"):
        """
        'loadFullModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. As opposed to the 'loadModel()' function, you don't need to specify the model type. This means you can load any Keras model trained with or without ImageAI and perform image prediction.
        - prediction_speed (optional), Acceptable values are "normal", "fast", "faster" and "fastest"

        :param prediction_speed:
        :return:
        """

        self.getJson()
                

        # Adjust the size that the input image will be rescaled to
        # Smaller numbers mean smaller picture which means quicker analysis and prediction
        # The trade-off to faster speeds is prediction accuracy
        match classification_speed:
            case 'normal':
                self.__input_image_size = 224
            case 'fast':
                self.__input_image_size = 160
            case 'faster':
                self.__input_image_size = 120
            case 'fastest':
                self.__input_image_size = 100


        # Only load model if a model hasn't already been loaded
        if (self.__modelLoaded == False):
            
            model = tf.keras.models.load_model(filepath=self.modelPath)
            self.__model_collection.append(model)
            self.__modelLoaded = True
            self.__modelType = "full"


    def classifyImage(self, image_input, result_count=3, input_type="file"):
        """
        'classifyImage()' function is used to classify a given image by receiving the following arguments:
            * input_type (optional) , the type of input to be parsed. Acceptable values are "file", "array" and "stream"
            * image_input , file path/numpy array/image file stream of the image.
            * result_count (optional) , the number of classifications to be sent which must be whole numbers between
                1 and 1000. The default is 5.

        This function returns 2 arrays namely 'classification_results' and 'classification_probabilities'. The 'classification_results'
        contains possible objects classes arranged in descending of their percentage probabilities. The 'classification_probabilities'
        contains the percentage probability of each object class. The position of each object class in the 'classification_results'
        array corresponds with the positions of the percentage probability in the 'classification_probabilities' array.


        :param input_type:
        :param image_input:
        :param result_count:
        :return classification_results, classification_probabilities:
        """
        classification_results = []
        classification_probabilities = []
        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making classification.")

        else:
            if (input_type == "file"):
                try:
                    image_to_predict = tf.keras.utils.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
                    image_to_predict = tf.keras.utils.img_to_array(image_to_predict)
                    image_to_predict = tf.expand_dims(image_to_predict, 0)
                except:
                    raise ValueError("You have set a path to an invalid image file.")
            elif (input_type == "array"):
                try:
                    image_input = tf.keras.utils.array_to_image(np.uint8(image_input))
                    image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                    image_input = tf.expand_dims(image_input, 0)
                    image_to_predict = image_input.copy()
                    image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                except:
                    raise ValueError("You have parsed in a wrong numpy array for the image")
            elif (input_type == "stream"):
                try:
                    image_input = Image.open(image_input)
                    image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                    image_input = tf.expand_dims(image_input, axis=0)
                    image_to_predict = image_input.copy()
                    image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                    
                except:
                    raise ValueError("You have parsed in a wrong stream for the image")

            # Preprocess image
            # if (self.__modelType == "MobileNetV2"):
            #     image_to_predict = tf.keras.applications.mobilenet_v2.preprocess_input(image_to_predict)
            # elif (self.__modelType == "ResNet50"):
            #     image_to_predict = tf.keras.applications.resnet50.preprocess_input(image_to_predict)
            # elif (self.__modelType == "InceptionV3"):
            #     image_to_predict = tf.keras.applications.inception_v3.preprocess_input(image_to_predict)
            # elif (self.__modelType == "DenseNet121"):
            #     image_to_predict = tf.keras.applications.densenet.preprocess_input(image_to_predict)
            try:
                model = self.__model_collection[0]
                prediction = model.predict(image_to_predict)

                predictiondata = []
                for pred in prediction:
                    scores = tf.nn.softmax(pred)
                    top_indices = pred.argsort()[-result_count:][::-1]
                    for i in top_indices:
                        each_result = []
                        each_result.append(self.__model_classes[str(i)])
                        each_result.append(100 * scores[i].numpy())
                        predictiondata.append(each_result)

                for result in predictiondata:
                    classification_results.append(str(result[0]))
                    classification_probabilities.append(str(result[1]))
                        
            except:
                raise ValueError("Error. Ensure your input image is valid")

            return classification_results, classification_probabilities
                


    def setModelTypeAsSqueezeNet(self):
        raise ValueError("ImageAI no longer support SqueezeNet. You can use MobileNetV2 instead by downloading the MobileNetV2 model and call the function 'setModelTypeAsMobileNetV2'")

    @deprecated(since="2.1.6", message="'.predictImage()' has been deprecated! Please use 'classifyImage()' instead.")
    def predictImage(self, image_input, result_count=3, input_type="file"):
        return self.classifyImage(image_input, result_count, input_type)

    @deprecated(since="2.1.6", message="'.setModelTypeAsResNet()' has been deprecated! Please use 'setModelType('ResNet50')' instead.")
    def setModelTypeAsResNet(self):
        return self.setModelTypeAsResNet50()

    @deprecated(since="2.1.6", message="'.setModelTypeAsDenseNet()' has been deprecated! Please use 'setModelType('DenseNet121')' instead.")
    def setModelTypeAsDenseNet(self):
        return self.setModelType('DenseNet121')

    @deprecated(since="2.2.0", message="'.setModelTypeAsMobileNetV2()' has been deprecated! Please use 'setModelType('MobileNetV2')' instead.")
    def setModelTypeAsMobileNetV2(self):
        return self.setModelType('MobileNetV2')

    @deprecated(since="2.2.0", message="'.setModelTypeAsResNet50()' has been deprecated! Please use 'setModelType('ResNet50')' instead.")
    def setModelTypeAsResNet50(self):
        return self.setModelType('ResNet50')

    @deprecated(since="2.2.0", message="'.setModelTypeAsDenseNet121()' has been deprecated! Please use 'setModelType('DenseNet121')' instead.")
    def setModelTypeAsDenseNet121(self):
        return self.setModelType('DenseNet121')

    @deprecated(since="2.2.0", message="'.setModelTypeAsInceptionV3()' has been deprecated! Please use 'setModelType('InceptionV3')' instead.")
    def setModelTypeAsInceptionV3(self):
        return self.setModelType('InceptionV3')