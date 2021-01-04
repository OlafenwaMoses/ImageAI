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
        self.__train_dir = ""
        self.__test_dir = ""
        self.__logs_dir = ""
        self.__num_epochs = 10
        self.__trained_model_dir = ""
        self.__model_class_dir = ""
        self.__initial_learning_rate = 1e-3
        self.__model_collection = []


    def setModelTypeAsSqueezeNet(self):
        raise ValueError("ImageAI no longer support SqueezeNet. You can use MobileNetV2 instead by downloading the MobileNetV2 model and call the function 'setModelTypeAsMobileNetV2'")

    def setModelTypeAsMobileNetV2(self):
        """
        'setModelTypeAsMobileNetV2()' is used to set the model type to the MobileNetV2 model
        for the training instance object .
        :return:
        """
        self.__modelType = "mobilenetv2"

    @deprecated(since="2.1.6", message="'.setModelTypeAsResNet()' has been deprecated! Please use 'setModelTypeAsResNet50()' instead.")
    def setModelTypeAsResNet(self):
        return self.setModelTypeAsResNet50()

    def setModelTypeAsResNet50(self):
        """
         'setModelTypeAsResNet()' is used to set the model type to the ResNet model
                for the training instance object .
        :return:
        """
        self.__modelType = "resnet50"

    
    @deprecated(since="2.1.6", message="'.setModelTypeAsDenseNet()' has been deprecated! Please use 'setModelTypeAsDenseNet121()' instead.")
    def setModelTypeAsDenseNet(self):
        return self.setModelTypeAsDenseNet121()

    def setModelTypeAsDenseNet121(self):
        """
         'setModelTypeAsDenseNet()' is used to set the model type to the DenseNet model
                for the training instance object .
        :return:
        """
        self.__modelType = "densenet121"

    def setModelTypeAsInceptionV3(self):
        """
         'setModelTypeAsInceptionV3()' is used to set the model type to the InceptionV3 model
                for the training instance object .
        :return:
        """
        self.__modelType = "inceptionv3"

    def setDataDirectory(self, data_directory="", train_subdirectory="train", test_subdirectory="test",
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




    def trainModel(self, num_objects, num_experiments=200, enhance_data=False, batch_size = 32, initial_learning_rate=1e-3, show_network_summary=False, training_image_size = 224, continue_from_model=None, transfer_from_model=None, transfer_with_full_training=True, initial_num_objects = None, save_full_model = False):

        """
        'trainModel()' function starts the model actual training. It accepts the following values:
        - num_objects , which is the number of classes present in the dataset that is to be used for training
        - num_experiments , also known as epochs, it is the number of times the network will train on all the training dataset
        - enhance_data (optional) , this is used to modify the dataset and create more instance of the training set to enhance the training result
        - batch_size (optional) , due to memory constraints, the network trains on a batch at once, until all the training set is exhausted. The value is set to 32 by default, but can be increased or decreased depending on the meormory of the compute used for training. The batch_size is conventionally set to 16, 32, 64, 128.
        - initial_learning_rate(optional) , this value is used to adjust the weights generated in the network. You rae advised to keep this value as it is if you don't have deep understanding of this concept.
        - show_network_summary(optional) , this value is used to show the structure of the network should you desire to see it. It is set to False by default
        - training_image_size(optional) , this value is used to define the image size on which the model will be trained. The value is 224 by default and is kept at a minimum of 100.
        - continue_from_model (optional) , this is used to set the path to a model file trained on the same dataset. It is primarily for continuos training from a previously saved model.
        - transfer_from_model (optional) , this is used to set the path to a model file trained on another dataset. It is primarily used to perform tramsfer learning.
        - transfer_with_full_training (optional) , this is used to set the pre-trained model to be re-trained across all the layers or only at the top layers.
        - initial_num_objects (required if 'transfer_from_model' is set ), this is used to set the number of objects the model used for transfer learning is trained on. If 'transfer_from_model' is set, this must be set as well.
        - save_full_model ( optional ), this is used to save the trained models with their network types. Any model saved by this specification can be loaded without specifying the network type.


        :param num_objects:
        :param num_experiments:
        :param enhance_data:
        :param batch_size:
        :param initial_learning_rate:
        :param show_network_summary:
        :param training_image_size:
        :param continue_from_model:
        :param transfer_from_model:
        :param initial_num_objects:
        :param save_full_model:
        :return:
        """
        self.__num_epochs = num_experiments
        self.__initial_learning_rate = initial_learning_rate
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)


        if(training_image_size < 100):
            warnings.warn("The specified training_image_size {} is less than 100. Hence the training_image_size will default to 100.".format(training_image_size))
            training_image_size = 100



        if (self.__modelType == "mobilenetv2"):
            if (continue_from_model != None):
                model = tf.keras.applications.MobileNetV2(input_shape=(training_image_size, training_image_size, 3), weights=continue_from_model, classes=num_objects,
                include_top=True)
                if (show_network_summary == True):
                    print("Training using weights from a previouly model")
            elif (transfer_from_model != None):
                base_model = tf.keras.applications.MobileNetV2(input_shape=(training_image_size, training_image_size, 3), weights= transfer_from_model,
                include_top=False, pooling="avg")

                network = base_model.output
                network = tf.keras.layers.Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = tf.keras.model.Models(inputs=base_model.input, outputs=network)

                if (show_network_summary == True):
                    print("Training using weights from a pre-trained ImageNet model")
            else:
                base_model = tf.keras.applications.MobileNetV2(input_shape=(training_image_size, training_image_size, 3), weights= None, classes=num_objects,
                include_top=False, pooling="avg")
                
                network = base_model.output
                network = tf.keras.layers.Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = tf.keras.models.Model(inputs=base_model.input, outputs=network)

        elif (self.__modelType == "resnet50"):
            if (continue_from_model != None):
                model = tf.keras.applications.ResNet50(input_shape=(training_image_size, training_image_size, 3), weights=continue_from_model, classes=num_objects,
                include_top=True)
                if (show_network_summary == True):
                    print("Training using weights from a previouly model")
            elif (transfer_from_model != None):
                base_model = tf.keras.applications.ResNet50(input_shape=(training_image_size, training_image_size, 3), weights= transfer_from_model,
                include_top=False, pooling="avg")

                network = base_model.output
                network = tf.keras.layers.Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = tf.keras.model.Models(inputs=base_model.input, outputs=network)

                if (show_network_summary == True):
                    print("Training using weights from a pre-trained ImageNet model")
            else:
                base_model = tf.keras.applications.ResNet50(input_shape=(training_image_size, training_image_size, 3), weights= None, classes=num_objects,
                include_top=False, pooling="avg")

                network = base_model.output
                network = tf.keras.layers.Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = tf.keras.models.Model(inputs=base_model.input, outputs=network)

        elif (self.__modelType == "inceptionv3"):

            if (continue_from_model != None):
                model = tf.keras.applications.InceptionV3(input_shape=(training_image_size, training_image_size, 3), weights=continue_from_model, classes=num_objects,
                include_top=True)
                if (show_network_summary == True):
                    print("Training using weights from a previouly model")
            elif (transfer_from_model != None):
                base_model = tf.keras.applications.InceptionV3(input_shape=(training_image_size, training_image_size, 3), weights= transfer_from_model,
                include_top=False, pooling="avg")

                network = base_model.output
                network = tf.keras.layers.Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = tf.keras.model.Models(inputs=base_model.input, outputs=network)

                if (show_network_summary == True):
                    print("Training using weights from a pre-trained ImageNet model")
            else:
                base_model = tf.keras.applications.InceptionV3(input_shape=(training_image_size, training_image_size, 3), weights= None, classes=num_objects,
                include_top=False, pooling="avg")

                network = base_model.output
                network = tf.keras.layers.Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = tf.keras.models.Model(inputs=base_model.input, outputs=network)

            base_model = tf.keras.applications.InceptionV3(input_shape=(training_image_size, training_image_size, 3), weights= None, classes=num_objects,
                include_top=False, pooling="avg")

        elif (self.__modelType == "densenet121"):
            if (continue_from_model != None):
                model = tf.keras.applications.DenseNet121(input_shape=(training_image_size, training_image_size, 3), weights=continue_from_model, classes=num_objects,
                include_top=True)
                if (show_network_summary == True):
                    print("Training using weights from a previouly model")
            elif (transfer_from_model != None):
                base_model = tf.keras.applications.DenseNet121(input_shape=(training_image_size, training_image_size, 3), weights= transfer_from_model,
                include_top=False, pooling="avg")

                network = base_model.output
                network = tf.keras.layers.Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = tf.keras.model.Models(inputs=base_model.input, outputs=network)

                if (show_network_summary == True):
                    print("Training using weights from a pre-trained ImageNet model")
            else:
                base_model = tf.keras.applications.DenseNet121(input_shape=(training_image_size, training_image_size, 3), weights= None, classes=num_objects,
                include_top=False, pooling="avg")

                network = base_model.output
                network = tf.keras.layers.Dense(num_objects, activation='softmax',
                         use_bias=True)(network)
                
                model = tf.keras.models.Model(inputs=base_model.input, outputs=network)

            base_model = tf.keras.applications.DenseNet121(input_shape=(training_image_size, training_image_size, 3), weights= None, classes=num_objects,
                include_top=False, pooling="avg")


        optimizer = tf.keras.optimizers.Adam(lr=self.__initial_learning_rate, decay=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        if (show_network_summary == True):
            model.summary()

        model_name = 'model_ex-{epoch:03d}_acc-{accuracy:03f}.h5'

        log_name = '{}_lr-{}_{}'.format(self.__modelType, initial_learning_rate, time.strftime("%Y-%m-%d-%H-%M-%S"))

        if not os.path.isdir(self.__trained_model_dir):
            os.makedirs(self.__trained_model_dir)

        if not os.path.isdir(self.__model_class_dir):
            os.makedirs(self.__model_class_dir)

        if not os.path.isdir(self.__logs_dir):
            os.makedirs(self.__logs_dir)

        model_path = os.path.join(self.__trained_model_dir, model_name)


        logs_path = os.path.join(self.__logs_dir, log_name)
        if not os.path.isdir(logs_path):
            os.makedirs(logs_path)

        save_weights_condition = True

        if(save_full_model == True ):
            save_weights_condition = False


        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                     monitor='accuracy',
                                     verbose=1,
                                     save_weights_only=save_weights_condition,
                                     save_best_only=True,
                                     period=1)


        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, 
                                  histogram_freq=0, 
                                  write_graph=False, 
                                  write_images=False)
        

        if (enhance_data == True):
            print("Using Enhanced Data Generation")

        height_shift = 0
        width_shift = 0
        if (enhance_data == True):
            height_shift = 0.1
            width_shift = 0.1

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=enhance_data, height_shift_range=height_shift, width_shift_range=width_shift)

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(self.__train_dir, target_size=(training_image_size, training_image_size),
                                                            batch_size=batch_size,
                                                            class_mode="categorical")
        test_generator = test_datagen.flow_from_directory(self.__test_dir, target_size=(training_image_size, training_image_size),
                                                          batch_size=batch_size,
                                                          class_mode="categorical")

        class_indices = train_generator.class_indices
        class_json = {}
        for eachClass in class_indices:
            class_json[str(class_indices[eachClass])] = eachClass

        with open(os.path.join(self.__model_class_dir, "model_class.json"), "w+") as json_file:
            json.dump(class_json, json_file, indent=4, separators=(",", " : "),
                      ensure_ascii=True)
            json_file.close()
        print("JSON Mapping for the model classes saved to ", os.path.join(self.__model_class_dir, "model_class.json"))

        num_train = len(train_generator.filenames)
        num_test = len(test_generator.filenames)
        print("Number of experiments (Epochs) : ", self.__num_epochs)

        
        model.fit_generator(train_generator, steps_per_epoch=int(num_train / batch_size), epochs=self.__num_epochs,
                            validation_data=test_generator,
                            validation_steps=int(num_test / batch_size), callbacks=[checkpoint, lr_scheduler])





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
        'setJsonPath()'

        :param model_path:
        :return:
        """
        self.jsonPath = model_json

    def setModelTypeAsMobileNetV2(self):
        """
        'setModelTypeAsMobileNetV2()' is used to set the model type to the MobileNetV2 model
        for the classification instance object .
        :return:
        """
        self.__modelType = "mobilenetv2"

    def setModelTypeAsResNet50(self):
        """
         'setModelTypeAsResNet50()' is used to set the model type to the ResNet50 model
                for the classification instance object .
        :return:
        """
        self.__modelType = "resnet50"

    def setModelTypeAsDenseNet121(self):
        """
         'setModelTypeAsDenseNet121()' is used to set the model type to the DenseNet121 model
                for the classification instance object .
        :return:
        """
        self.__modelType = "densenet121"

    def setModelTypeAsInceptionV3(self):
        """
         'setModelTypeAsInceptionV3()' is used to set the model type to the InceptionV3 model
                for the classification instance object .
        :return:
        """
        self.__modelType = "inceptionv3"

    def loadModel(self, classification_speed="normal", num_objects=10):
        """
        'loadModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. This function receives an optional value which is "classification_speed".
        The value is used to reduce the time it takes to classify an image, down to about 50% of the normal time,
        with just slight changes or drop in classification accuracy, depending on the nature of the image.
        * classification_speed (optional); Acceptable values are "normal", "fast", "faster" and "fastest"

        :param classification_speed :
        :return:
        """

        self.__model_classes = json.load(open(self.jsonPath))

        if(classification_speed=="normal"):
            self.__input_image_size = 224
        elif(classification_speed=="fast"):
            self.__input_image_size = 160
        elif(classification_speed=="faster"):
            self.__input_image_size = 120
        elif (classification_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):

            image_input = tf.keras.layers.Input(shape=(self.__input_image_size, self.__input_image_size, 3))

            if(self.__modelType == "" ):
                raise ValueError("You must set a valid model type before loading the model.")

            elif(self.__modelType == "mobilenetv2"):
                model = tf.keras.applications.MobileNetV2(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=self.modelPath, classes = num_objects )
                self.__model_collection.append(model)
                self.__modelLoaded = True
                try:
                    None
                except:
                    raise ValueError("An error occured. Ensure your model file is a MobileNetV2 Model and is located in the path {}".format(self.modelPath))

            elif(self.__modelType == "resnet50"):
                try:
                    model = tf.keras.applications.ResNet50(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes = num_objects )
                    model.load_weights(self.modelPath)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is a ResNet50 Model and is located in the path {}".format(self.modelPath))

            elif (self.__modelType == "densenet121"):
                try:
                    model = tf.keras.applications.DenseNet121(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=self.modelPath, classes = num_objects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is a DenseNet121 Model and is located in the path {}".format(self.modelPath))

            elif (self.__modelType == "inceptionv3"):
                try:
                    model = tf.keras.applications.InceptionV3(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=self.modelPath, classes = num_objects )
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is in {}".format(self.modelPath))
    def loadFullModel(self, classification_speed="normal", num_objects=10):
        """
        'loadFullModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. As opposed to the 'loadModel()' function, you don't need to specify the model type. This means you can load any Keras model trained with or without ImageAI and perform image prediction.
        - prediction_speed (optional), Acceptable values are "normal", "fast", "faster" and "fastest"
        - num_objects (required), the number of objects the model is trained to recognize

        :param prediction_speed:
        :param num_objects:
        :return:
        """

        self.numObjects = num_objects
        self.__model_classes = json.load(open(self.jsonPath))

        if (classification_speed == "normal"):
            self.__input_image_size = 224
        elif (classification_speed == "fast"):
            self.__input_image_size = 160
        elif (classification_speed == "faster"):
            self.__input_image_size = 120
        elif (classification_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):
            
            model = tf.keras.models.load_model(filepath=self.modelPath)
            self.__model_collection.append(model)
            self.__modelLoaded = True
            self.__modelType = "full"

    def getModels(self):
        """
        'getModels()' provides access to the internal model collection. Helpful if models are used down the line with tools like lime.
        :return:
        """
        return self.__model_collection


    def classifyImage(self, image_input, result_count=5, input_type="file"):
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
                    image_to_predict = tf.keras.preprocessing.image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
                    image_to_predict = tf.keras.preprocessing.image.img_to_array(image_to_predict, data_format="channels_last")
                    image_to_predict = np.expand_dims(image_to_predict, axis=0)
                except:
                    raise ValueError("You have set a path to an invalid image file.")
            elif (input_type == "array"):
                try:
                    image_input = Image.fromarray(np.uint8(image_input))
                    image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                    image_input = np.expand_dims(image_input, axis=0)
                    image_to_predict = image_input.copy()
                    image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                except:
                    raise ValueError("You have parsed in a wrong numpy array for the image")
            elif (input_type == "stream"):
                try:
                    image_input = Image.open(image_input)
                    image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                    image_input = np.expand_dims(image_input, axis=0)
                    image_to_predict = image_input.copy()
                    image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                    
                except:
                    raise ValueError("You have parsed in a wrong stream for the image")

            if (self.__modelType == "mobilenetv2"):
                image_to_predict = tf.keras.applications.mobilenet_v2.preprocess_input(image_to_predict)
            elif (self.__modelType == "full"):
                image_to_predict = tf.keras.applications.mobilenet_v2.preprocess_input(image_to_predict)
            elif (self.__modelType == "inceptionv3"):
                image_to_predict = tf.keras.applications.inception_v3.preprocess_input(image_to_predict)
            elif (self.__modelType == "densenet121"):
                image_to_predict = tf.keras.applications.densenet.preprocess_input(image_to_predict)
            try:
                model = self.__model_collection[0]
                prediction = model.predict(image_to_predict, steps=1)

                predictiondata = []
                for pred in prediction:
                    top_indices = pred.argsort()[-result_count:][::-1]
                    for i in top_indices:
                        each_result = []
                        each_result.append(self.__model_classes[str(i)])
                        each_result.append(pred[i])
                        predictiondata.append(each_result)

                for result in predictiondata:
                    classification_results.append(str(result[0]))
                    classification_probabilities.append(result[1] * 100)
                        
            except:
                raise ValueError("Error. Ensure your input image is valid")

            return classification_results, classification_probabilities
                

    @deprecated(since="2.1.6", message="'.predictImage()' has been deprecated! Please use 'classifyImage()' instead.")
    def predictImage(self, image_input, result_count=5, input_type="file"):

        return self.classifyImage(image_input, result_count, input_type)