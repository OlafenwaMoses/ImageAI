from ..SqueezeNet.squeezenet import SqueezeNet
from ..ResNet.resnet50 import ResNet50
from ..InceptionV3.inceptionv3 import InceptionV3
from ..DenseNet.densenet import DenseNetImageNet121
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Flatten, Dense, Input, Conv2D, GlobalAvgPool2D, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from PIL import Image
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint
from io import open
import json
import numpy as np
import warnings





class ModelTraining:
    """
        This is the Model training class, that allows you to define a deep learning network
        from the 4 available networks types supported by ImageAI which are SqueezeNet, ResNet50,
        InceptionV3 and DenseNet121. Once you instantiate this class, you must call:

        *
    """

    def __init__(self):
        self.__modelType = ""
        self.__use_pretrained_model = False
        self.__data_dir = ""
        self.__train_dir = ""
        self.__test_dir = ""
        self.__num_epochs = 10
        self.__trained_model_dir = ""
        self.__model_class_dir = ""
        self.__initial_learning_rate = 1e-3
        self.__model_collection = []




    def setModelTypeAsSqueezeNet(self):
        """
        'setModelTypeAsSqueezeNet()' is used to set the model type to the SqueezeNet model
        for the training instance object .
        :return:
        """
        self.__modelType = "squeezenet"

    def setModelTypeAsResNet(self):
        """
         'setModelTypeAsResNet()' is used to set the model type to the ResNet model
                for the training instance object .
        :return:
        """
        self.__modelType = "resnet"

    def setModelTypeAsDenseNet(self):
        """
         'setModelTypeAsDenseNet()' is used to set the model type to the DenseNet model
                for the training instance object .
        :return:
        """
        self.__modelType = "densenet"

    def setModelTypeAsInceptionV3(self):
        """
         'setModelTypeAsInceptionV3()' is used to set the model type to the InceptionV3 model
                for the training instance object .
        :return:
        """
        self.__modelType = "inceptionv3"

    def setDataDirectory(self, data_directory=""):
        """
                 'setDataDirectory()' is required to set the path to which the data/dataset to be used for
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

                :return:
                """

        self.__data_dir = data_directory
        self.__train_dir = os.path.join(self.__data_dir, "train")
        self.__test_dir = os.path.join(self.__data_dir, "test")
        self.__trained_model_dir = os.path.join(self.__data_dir, "models")
        self.__model_class_dir = os.path.join(self.__data_dir, "json")

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




    def trainModel(self, num_objects, num_experiments=200, enhance_data=False, batch_size = 32, initial_learning_rate=1e-3, show_network_summary=False, training_image_size = 224):

        """
                 'trainModel()' function starts the actual training. It accepts the following values:
                 - num_objects , which is the number of classes present in the dataset that is to be used for training
                 - num_experiments , also known as epochs, it is the number of times the network will train on all the training dataset
                 - enhance_data (optional) , this is used to modify the dataset and create more instance of the training set to enhance the training result
                 - batch_size (optional) , due to memory constraints, the network trains on a batch at once, until all the training set is exhausted.
                                            The value is set to 32 by default, but can be increased or decreased depending on the meormory of the
                                            compute used for training. The batch_size is conventionally set to 16, 32, 64, 128.
                 - initial_learning_rate(optional) , this value is used to adjust the weights generated in the network. You rae advised
                                                     to keep this value as it is if you don't have deep understanding of this concept.
                 - show_network_summary(optional) , this value is used to show the structure of the network should you desire to see it.
                                                    Itis set to False by default
                 - training_image_size(optional) , this value is used to define the image size on which the model will be trained. The
                                            value is 224 by default and is kept at a minimum of 100.

                 *

                :param num_objects:
                :param num_experiments:
                :param enhance_data:
                :param batch_size:
                :param initial_learning_rate:
                :param show_network_summary:
                :param training_image_size:
                :return:
                """
        self.__num_epochs = num_experiments
        self.__initial_learning_rate = initial_learning_rate
        lr_scheduler = LearningRateScheduler(self.lr_schedule)


        num_classes = num_objects

        if(training_image_size < 100):
            warnings.warn("The specified training_image_size {} is less than 100. Hence the training_image_size will default to 100.".format(training_image_size))
            training_image_size = 100



        image_input = Input(shape=(training_image_size, training_image_size, 3))
        if (self.__modelType == "squeezenet"):
            model = SqueezeNet(weights="custom", num_classes=num_classes, model_input=image_input)
        elif (self.__modelType == "resnet"):
            model = ResNet50(weights="custom", num_classes=num_classes, model_input=image_input)
        elif (self.__modelType == "inceptionv3"):
            model = InceptionV3(weights="custom", classes=num_classes, model_input=image_input)
        elif (self.__modelType == "densenet"):
            model = DenseNetImageNet121(weights="custom", classes=num_classes, model_input=image_input)

        optimizer = Adam(lr=self.__initial_learning_rate, decay=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        if (show_network_summary == True):
            model.summary()

        model_name = 'model_ex-{epoch:03d}_acc-{val_acc:03f}.h5'

        if not os.path.isdir(self.__trained_model_dir):
            os.makedirs(self.__trained_model_dir)

        if not os.path.isdir(self.__model_class_dir):
            os.makedirs(self.__model_class_dir)

        model_path = os.path.join(self.__trained_model_dir, model_name)

        checkpoint = ModelCheckpoint(filepath=model_path,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_weights_only=True,
                                     period=1)

        if (enhance_data == True):
            print("Using Enhanced Data Generation")

        height_shift = 0
        width_shift = 0
        if (enhance_data == True):
            height_shift = 0.1
            width_shift = 0.1

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=enhance_data, height_shift_range=height_shift, width_shift_range=width_shift)

        test_datagen = ImageDataGenerator(
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

        #
        model.fit_generator(train_generator, steps_per_epoch=int(num_train / batch_size), epochs=self.__num_epochs,
                            validation_data=test_generator,
                            validation_steps=int(num_test / batch_size), callbacks=[checkpoint, lr_scheduler])





class CustomImagePrediction:
    """
                This is the image prediction class for custom models trained with the 'ModelTraining' class. It provides support for 4 different models which are:
                 ResNet50, SqueezeNet, DenseNet121 and Inception V3. After instantiating this class, you can set it's properties and
                 make image predictions using it's pre-defined functions.

                 The following functions are required to be called before a prediction can be made
                 * setModelPath() , path to your custom model
                 * setJsonPath , , path to your custom model's corresponding JSON file
                 * At least of of the following and it must correspond to the model set in the setModelPath()
                  [setModelTypeAsSqueezeNet(), setModelTypeAsResNet(), setModelTypeAsDenseNet, setModelTypeAsInceptionV3]
                 * loadModel() [This must be called once only before making a prediction]

                 Once the above functions have been called, you can call the predictImage() function of the prediction instance
                 object at anytime to predict an image.
        """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.jsonPath = ""
        self.numObjects = 10
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_size = 224

    def setModelPath(self, model_path):
        """
        'setModelPath()' function is required and is used to set the file path to the model adopted from the list of the
        available 4 model types. The model path must correspond to the model type set for the prediction instance object.

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

    def setModelTypeAsSqueezeNet(self):
        """
        'setModelTypeAsSqueezeNet()' is used to set the model type to the SqueezeNet model
        for the prediction instance object .
        :return:
        """
        self.__modelType = "squeezenet"

    def setModelTypeAsResNet(self):
        """
         'setModelTypeAsResNet()' is used to set the model type to the ResNet model
                for the prediction instance object .
        :return:
        """
        self.__modelType = "resnet"

    def setModelTypeAsDenseNet(self):
        """
         'setModelTypeAsDenseNet()' is used to set the model type to the DenseNet model
                for the prediction instance object .
        :return:
        """
        self.__modelType = "densenet"

    def setModelTypeAsInceptionV3(self):
        """
         'setModelTypeAsInceptionV3()' is used to set the model type to the InceptionV3 model
                for the prediction instance object .
        :return:
        """
        self.__modelType = "inceptionv3"

    def loadModel(self, prediction_speed="normal", num_objects=10):
        """
        'loadModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. This function receives an optional value which is "prediction_speed".
        The value is used to reduce the time it takes to predict an image, down to about 50% of the normal time,
        with just slight changes or drop in prediction accuracy, depending on the nature of the image.
        * prediction_speed (optional); Acceptable values are "normal", "fast", "faster" and "fastest"

        :param prediction_speed :
        :return:
        """

        self.numObjects = num_objects

        if (prediction_speed == "normal"):
            self.__input_image_size = 224
        elif (prediction_speed == "fast"):
            self.__input_image_size = 160
        elif (prediction_speed == "faster"):
            self.__input_image_size = 120
        elif (prediction_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):

            image_input = Input(shape=(self.__input_image_size, self.__input_image_size, 3))

            if (self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")


            elif (self.__modelType == "squeezenet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from ..SqueezeNet.squeezenet import SqueezeNet
                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions

                model = SqueezeNet(model_path=self.modelPath, weights="trained", model_input=image_input,
                                   num_classes=self.numObjects)
                self.__model_collection.append(model)
                self.__modelLoaded = True
                try:
                    None
                except:
                    raise ("You have specified an incorrect path to the SqueezeNet model file.")
            elif (self.__modelType == "resnet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from ..ResNet.resnet50 import ResNet50
                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions
                try:
                    model = ResNet50(model_path=self.modelPath, weights="trained", model_input=image_input, num_classes=self.numObjects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the ResNet model file.")

            elif (self.__modelType == "densenet"):
                from tensorflow.python.keras.preprocessing import image
                from ..DenseNet.densenet import DenseNetImageNet121
                from .custom_utils import decode_predictions, preprocess_input
                import numpy as np
                try:
                    model = DenseNetImageNet121(model_path=self.modelPath, weights="trained", model_input=image_input, classes=self.numObjects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the DenseNet model file.")

            elif (self.__modelType == "inceptionv3"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                from .custom_utils import decode_predictions, preprocess_input



                try:
                    model = InceptionV3(include_top=True, weights="trained", model_path=self.modelPath,
                                        model_input=image_input, classes=self.numObjects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the InceptionV3 model file.")

    def predictImage(self, image_input, result_count=1, input_type="file"):
        """
        'predictImage()' function is used to predict a given image by receiving the following arguments:
            * input_type (optional) , the type of input to be parsed. Acceptable values are "file", "array" and "stream"
            * image_input , file path/numpy array/image file stream of the image.
            * result_count (optional) , the number of predictions to be sent which must be whole numbers between
                1 and the number of classes present in the model

        This function returns 2 arrays namely 'prediction_results' and 'prediction_probabilities'. The 'prediction_results'
        contains possible objects classes arranged in descending of their percentage probabilities. The 'prediction_probabilities'
        contains the percentage probability of each object class. The position of each object class in the 'prediction_results'
        array corresponds with the positions of the percentage possibilities in the 'prediction_probabilities' array.


        :param input_type:
        :param image_input:
        :param result_count:
        :return prediction_results, prediction_probabilities:
        """
        prediction_results = []
        prediction_probabilities = []
        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making predictions.")

        else:

            if (self.__modelType == "squeezenet"):

                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                model = self.__model_collection[0]

                prediction = model.predict(image_to_predict, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "resnet"):

                model = self.__model_collection[0]

                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                prediction = model.predict(x=image_to_predict, steps=1)




                try:

                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))


                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "densenet"):

                model = self.__model_collection[0]

                from .custom_utils import preprocess_input
                from .custom_utils import decode_predictions
                from ..DenseNet.densenet import DenseNetImageNet121
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                prediction = model.predict(x=image_to_predict, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "inceptionv3"):

                model = self.__model_collection[0]

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                from .custom_utils import decode_predictions, preprocess_input

                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                prediction = model.predict(x=image_to_predict, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities




    def predictMultipleImages(self, sent_images_array, result_count_per_image=1, input_type="file"):
        """
                'predictMultipleImages()' function is used to predict more than one image by receiving the following arguments:
                    * input_type , the type of inputs contained in the parsed array. Acceptable values are "file", "array" and "stream"
                    * sent_images_array , an array of image file paths, image numpy array or image file stream
                    * result_count_per_image (optionally) , the number of predictions to be sent per image, which must be whole numbers between
                        1 and the number of classes present in the model

                This function returns an array of dictionaries, with each dictionary containing 2 arrays namely 'prediction_results' and 'prediction_probabilities'. The 'prediction_results'
                contains possible objects classes arranged in descending of their percentage probabilities. The 'prediction_probabilities'
                contains the percentage probability of each object class. The position of each object class in the 'prediction_results'
                array corresponds with the positions of the percentage possibilities in the 'prediction_probabilities' array.


                :param input_type:
                :param sent_images_array:
                :param result_count_per_image:
                :return output_array:
                """

        output_array = []

        for image_input in sent_images_array:

            prediction_results = []
            prediction_probabilities = []
            if (self.__modelLoaded == False):
                raise ValueError("You must call the loadModel() function before making predictions.")

            else:
                if (self.__modelType == "squeezenet"):

                    from .custom_utils import preprocess_input
                    from .custom_utils import decode_predictions
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(
                                self.__input_image_size, self.__input_image_size))
                            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                            image_to_predict = np.expand_dims(image_to_predict, axis=0)

                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have set a path to an invalid image file.")
                    elif (input_type == "array"):
                        try:
                            image_input = Image.fromarray(np.uint8(image_input))
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong numpy array for the image")
                    elif (input_type == "stream"):
                        try:
                            image_input = Image.open(image_input)
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong stream for the image")

                    model = self.__model_collection[0]

                    prediction = model.predict(image_to_predict, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image), model_json=self.jsonPath)

                        for result in predictiondata:
                            prediction_results.append(str(result[0]))
                            prediction_probabilities.append(str(result[1] * 100))
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)

                elif (self.__modelType == "resnet"):

                    model = self.__model_collection[0]

                    from .custom_utils import preprocess_input
                    from .custom_utils import decode_predictions
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(
                                self.__input_image_size, self.__input_image_size))
                            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                            image_to_predict = np.expand_dims(image_to_predict, axis=0)

                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have set a path to an invalid image file.")
                    elif (input_type == "array"):
                        try:
                            image_input = Image.fromarray(np.uint8(image_input))
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong numpy array for the image")
                    elif (input_type == "stream"):
                        try:
                            image_input = Image.open(image_input)
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong stream for the image")

                    prediction = model.predict(x=image_to_predict, steps=1)

                    try:

                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image), model_json=self.jsonPath)

                        for result in predictiondata:
                            prediction_results.append(str(result[0]))
                            prediction_probabilities.append(str(result[1] * 100))


                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)


                elif (self.__modelType == "densenet"):

                    model = self.__model_collection[0]

                    from .custom_utils import preprocess_input
                    from .custom_utils import decode_predictions
                    from ..DenseNet.densenet import DenseNetImageNet121
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(
                                self.__input_image_size, self.__input_image_size))
                            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                            image_to_predict = np.expand_dims(image_to_predict, axis=0)

                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have set a path to an invalid image file.")
                    elif (input_type == "array"):
                        try:
                            image_input = Image.fromarray(np.uint8(image_input))
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong numpy array for the image")
                    elif (input_type == "stream"):
                        try:
                            image_input = Image.open(image_input)
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong stream for the image")

                    prediction = model.predict(x=image_to_predict, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image), model_json=self.jsonPath)

                        for result in predictiondata:
                            prediction_results.append(str(result[0]))
                            prediction_probabilities.append(str(result[1] * 100))
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)


                elif (self.__modelType == "inceptionv3"):

                    model = self.__model_collection[0]

                    from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                    from .custom_utils import decode_predictions, preprocess_input

                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(
                                self.__input_image_size, self.__input_image_size))
                            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                            image_to_predict = np.expand_dims(image_to_predict, axis=0)

                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have set a path to an invalid image file.")
                    elif (input_type == "array"):
                        try:
                            image_input = Image.fromarray(np.uint8(image_input))
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong numpy array for the image")
                    elif (input_type == "stream"):
                        try:
                            image_input = Image.open(image_input)
                            image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                            image_input = np.expand_dims(image_input, axis=0)
                            image_to_predict = image_input.copy()
                            image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                            image_to_predict = preprocess_input(image_to_predict)
                        except:
                            raise ValueError("You have parsed in a wrong stream for the image")

                    prediction = model.predict(x=image_to_predict, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image), model_json=self.jsonPath)

                        for result in predictiondata:
                            prediction_results.append(str(result[0]))
                            prediction_probabilities.append(str(result[1] * 100))
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)


        return output_array



