import numpy as np
from tensorflow.python.keras.preprocessing import image
from PIL import Image


from tensorflow.python.keras.layers import Input, Conv2D, MaxPool2D, Activation, concatenate, Dropout
from tensorflow.python.keras.layers import GlobalAvgPool2D, GlobalMaxPool2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential


class ImagePrediction:
    """
            This is the image prediction class in the ImageAI library. It provides support for 4 different models which are:
             ResNet, SqueezeNet, DenseNet and Inception V3. After instantiating this class, you can set it's properties and
             make image predictions using it's pre-defined functions.

             The following functions are required to be called before a prediction can be made
             * setModelPath()
             * At least of of the following and it must correspond to the model set in the setModelPath()
              [setModelTypeAsSqueezeNet(), setModelTypeAsResNet(), setModelTypeAsDenseNet, setModelTypeAsInceptionV3]
             * loadModel() [This must be called once only before making a prediction]

             Once the above functions have been called, you can call the predictImage() function of the prediction instance
             object at anytime to predict an image.
    """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
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

    def loadModel(self, prediction_speed="normal"):
        """
        'loadModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. This function receives an optional value which is "prediction_speed".
        The value is used to reduce the time it takes to predict an image, down to about 50% of the normal time,
        with just slight changes or drop in prediction accuracy, depending on the nature of the image.
        * prediction_speed (optional); Acceptable values are "normal", "fast", "faster" and "fastest"

        :param prediction_speed :
        :return:
        """

        if(prediction_speed=="normal"):
            self.__input_image_size = 224
        elif(prediction_speed=="fast"):
            self.__input_image_size = 160
        elif(prediction_speed=="faster"):
            self.__input_image_size = 120
        elif (prediction_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):

            image_input = Input(shape=(self.__input_image_size, self.__input_image_size, 3))

            if(self.__modelType == "" ):
                raise ValueError("You must set a valid model type before loading the model.")


            elif(self.__modelType == "squeezenet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from .SqueezeNet.squeezenet import SqueezeNet
                from .imagenet_utils import preprocess_input, decode_predictions
                try:
                    model = SqueezeNet(model_path=self.modelPath, model_input=image_input)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ("You have specified an incorrect path to the SqueezeNet model file.")
            elif(self.__modelType == "resnet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from .ResNet.resnet50 import ResNet50
                from .imagenet_utils import preprocess_input, decode_predictions
                try:
                    model = ResNet50(model_path=self.modelPath, model_input=image_input)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the ResNet model file.")

            elif (self.__modelType == "densenet"):
                from tensorflow.python.keras.preprocessing import image
                from .DenseNet.densenet import DenseNetImageNet121, preprocess_input, decode_predictions
                import numpy as np
                try:
                    model = DenseNetImageNet121(model_path=self.modelPath, model_input=image_input)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the DenseNet model file.")

            elif (self.__modelType == "inceptionv3"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                from imageai.Prediction.InceptionV3.inceptionv3 import preprocess_input, decode_predictions

                try:
                    model = InceptionV3(include_top=True, weights="imagenet", model_path=self.modelPath, model_input=image_input)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the InceptionV3 model file.")

                





            
    def predictImage(self, image_input, result_count=5, input_type="file" ):
        """
        'predictImage()' function is used to predict a given image by receiving the following arguments:
            * input_type (optional) , the type of input to be parsed. Acceptable values are "file", "array" and "stream"
            * image_input , file path/numpy array/image file stream of the image.
            * result_count (optional) , the number of predictions to be sent which must be whole numbers between
                1 and 1000. The default is 5.

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

                from .imagenet_utils import preprocess_input, decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(result[2] * 100)
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "resnet"):

                model = self.__model_collection[0]

                from .imagenet_utils import preprocess_input, decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(result[2] * 100)
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "densenet"):

                model = self.__model_collection[0]

                from .DenseNet.densenet import preprocess_input, decode_predictions
                from .DenseNet.densenet import DenseNetImageNet121
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(result[2] * 100)
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "inceptionv3"):

                model = self.__model_collection[0]

                from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3, preprocess_input, decode_predictions

                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(result[2] * 100)
                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities



    def predictMultipleImages(self, sent_images_array, result_count_per_image=2, input_type="file"):
        """
                'predictMultipleImages()' function is used to predict more than one image by receiving the following arguments:
                    * input_type , the type of inputs contained in the parsed array. Acceptable values are "file", "array" and "stream"
                    * sent_images_array , an array of image file paths, image numpy array or image file stream
                    * result_count_per_image (optionally) , the number of predictions to be sent per image, which must be whole numbers between
                        1 and 1000. The default is 2.

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

                    from .imagenet_utils import preprocess_input, decode_predictions
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(result[2] * 100)
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)

                elif (self.__modelType == "resnet"):

                    model = self.__model_collection[0]

                    from .imagenet_utils import preprocess_input, decode_predictions
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(result[2] * 100)
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)

                elif (self.__modelType == "densenet"):

                    model = self.__model_collection[0]

                    from .DenseNet.densenet import preprocess_input, decode_predictions
                    from .DenseNet.densenet import DenseNetImageNet121
                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(result[2] * 100)
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)

                elif (self.__modelType == "inceptionv3"):

                    model = self.__model_collection[0]

                    from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3, preprocess_input, \
                        decode_predictions

                    if (input_type == "file"):
                        try:
                            image_to_predict = image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
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
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(result[2] * 100)
                    except:
                        raise ValueError("An error occured! Try again.")

                    each_image_details = {}
                    each_image_details["predictions"] = prediction_results
                    each_image_details["percentage_probabilities"] = prediction_probabilities
                    output_array.append(each_image_details)


        return output_array

