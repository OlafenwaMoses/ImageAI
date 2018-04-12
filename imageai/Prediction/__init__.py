import numpy as np
from tensorflow.python.keras.preprocessing import image


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

    def loadModel(self):
        """
        'loadModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function.
        :return:
        """
        if (self.__modelLoaded == False):

            if(self.__modelType == "" ):
                raise ValueError("You must set a valid model type before loading the model.")


            elif(self.__modelType == "squeezenet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from .SqueezeNet.squeezenet import SqueezeNet
                from .SqueezeNet.imagenet_utils import preprocess_input, decode_predictions
                try:
                    model = SqueezeNet(model_path=self.modelPath)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ("You have specified an incorrect path to the SqueezeNet model file.")
            elif(self.__modelType == "resnet"):
                import numpy as np
                from tensorflow.python.keras.preprocessing import image
                from .ResNet.resnet50 import ResNet50
                from .ResNet.imagenet_utils import preprocess_input, decode_predictions
                try:
                    model = ResNet50(model_path=self.modelPath)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the ResNet model file.")

            elif (self.__modelType == "densenet"):
                from tensorflow.python.keras.preprocessing import image
                from .DenseNet.densenet import DenseNetImageNet121, preprocess_input, decode_predictions
                import numpy as np
                from .DenseNet.densenet import DenseNetImageNet121
                try:
                    model = DenseNetImageNet121(input_shape=(224, 224, 3), model_path=self.modelPath)
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
                    model = InceptionV3(include_top=True, weights="imagenet", model_path=self.modelPath)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the InceptionV3 model file.")

                





            
    def predictImage(self, image_path, result_count=5):
        """
        'predictImage()' function is used to predict a given image by receiving the following arguments:
            * image_path , file path to the image
            * result_count (optionally) , the number of predictions to be sent which must be whole numbers between
                1 and 1000. The default is 5.

        This function returns 2 arrays namely 'prediction_results' and 'prediction_probabilities'. The 'prediction_results'
        contains possible objects classes arranged in descending of their percentage probabilities. The 'prediction_probabilities'
        contains the percentage probability of each object class. The position of each object class in the 'prediction_results'
        array corresponds with the positions of the percentage possibilities in the 'prediction_probabilities' array.


        :param image_path:
        :param result_count:
        :return prediction_results, prediction_probabilities:
        """
        prediction_results = []
        prediction_probabilities = []
        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making predictions.")

        else:

            if (self.__modelType == "squeezenet"):

                try:
                    from .SqueezeNet.imagenet_utils import preprocess_input, decode_predictions
                    img = image.load_img(image_path, target_size=(227, 227))
                    img = image.img_to_array(img, data_format="channels_last")
                    img = np.expand_dims(img, axis=0)

                    img = preprocess_input(img, data_format="channels_last")
                except:
                    raise ValueError("You have set a path to an invalid image file.")

                model = self.__model_collection[0]

                prediction = model.predict(img, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(str(result[2] * 100))
                except:
                    raise ValueError("You have set a wrong path to the JSON file")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "resnet"):

                model = self.__model_collection[0]

                try:
                    from .ResNet.imagenet_utils import preprocess_input, decode_predictions
                    target_image = image.load_img(image_path, grayscale=False, target_size=(224, 224))
                    target_image = image.img_to_array(target_image, data_format="channels_last")
                    target_image = np.expand_dims(target_image, axis=0)

                    target_image = preprocess_input(target_image, data_format="channels_last")
                except:
                    raise ValueError("You have set a path to an invalid image file.")

                prediction = model.predict(x=target_image, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(str(result[2] * 100))
                except:
                    raise ValueError("You have set a wrong path to the JSON file")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "densenet"):

                model = self.__model_collection[0]

                try:
                    from .DenseNet.densenet import DenseNetImageNet121, preprocess_input, decode_predictions
                    from .DenseNet.densenet import DenseNetImageNet121
                    image_to_predict = image.load_img(image_path,
                                                      grayscale=False, target_size=(224, 224))
                    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                    image_to_predict = np.expand_dims(image_to_predict, axis=0)

                    image_to_predict = preprocess_input(image_to_predict, data_format="channels_last")
                except:
                    raise ValueError("You have set a path to an invalid image file.")

                prediction = model.predict(x=image_to_predict, steps=1)

                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(str(result[2] * 100))
                except:
                    raise ValueError("You have set a wrong path to the JSON file")

                return prediction_results, prediction_probabilities
            elif (self.__modelType == "inceptionv3"):

                model = self.__model_collection[0]

                try:
                    from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                    from imageai.Prediction.InceptionV3.inceptionv3 import preprocess_input, decode_predictions

                    image_to_predict = image.load_img(image_path,
                                                      grayscale=False, target_size=(299, 299))
                    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                    image_to_predict = np.expand_dims(image_to_predict, axis=0)

                    image_to_predict = preprocess_input(image_to_predict)
                except:
                    raise ValueError("You have set a path to an invalid image file.")

                prediction = model.predict(x=image_to_predict, steps=1)


                try:
                    predictiondata = decode_predictions(prediction, top=int(result_count))

                    for results in predictiondata:
                        countdown = 0
                        for result in results:
                            countdown += 1
                            prediction_results.append(str(result[1]))
                            prediction_probabilities.append(str(result[2] * 100))
                except:
                    raise ValueError("You have set a wrong path to the JSON file")

                return prediction_results, prediction_probabilities



    def predictMultipleImages(self, sent_images_array, result_count_per_image=2):
        """
                'predictMultipleImages()' function is used to predict more than one image by receiving the following arguments:
                    * sent_images_array , an array of image file paths
                    * result_count_per_image (optionally) , the number of predictions to be sent per image, which must be whole numbers between
                        1 and 1000. The default is 2.

                This function returns an array of dictionaries, with each dictionary containing 2 arrays namely 'prediction_results' and 'prediction_probabilities'. The 'prediction_results'
                contains possible objects classes arranged in descending of their percentage probabilities. The 'prediction_probabilities'
                contains the percentage probability of each object class. The position of each object class in the 'prediction_results'
                array corresponds with the positions of the percentage possibilities in the 'prediction_probabilities' array.


                :param sent_images_array:
                :param result_count_per_image:
                :return output_array:
                """

        output_array = []

        for eachImage in sent_images_array:

            prediction_results = []
            prediction_probabilities = []
            if (self.__modelLoaded == False):
                raise ValueError("You must call the loadModel() function before making predictions.")

            else:

                if (self.__modelType == "squeezenet"):

                    try:
                        from .SqueezeNet.imagenet_utils import preprocess_input, decode_predictions
                        img = image.load_img(eachImage, target_size=(227, 227))
                        img = image.img_to_array(img, data_format="channels_last")
                        img = np.expand_dims(img, axis=0)

                        img = preprocess_input(img, data_format="channels_last")
                    except:
                        raise ValueError("You have set a path to an invalid image file.")

                    model = self.__model_collection[0]

                    prediction = model.predict(img, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(str(result[2] * 100))

                        each_image_details = {}
                        each_image_details["predictions"] = prediction_results
                        each_image_details["percentage_probabilities"] = prediction_probabilities
                        output_array.append(each_image_details)
                    except:
                        raise ValueError("You have set a wrong path to the JSON file")


                elif (self.__modelType == "resnet"):

                    model = self.__model_collection[0]

                    try:
                        from .ResNet.imagenet_utils import preprocess_input, decode_predictions
                        target_image = image.load_img(eachImage, grayscale=False, target_size=(224, 224))
                        target_image = image.img_to_array(target_image, data_format="channels_last")
                        target_image = np.expand_dims(target_image, axis=0)

                        target_image = preprocess_input(target_image, data_format="channels_last")
                    except:
                        raise ValueError("You have set a path to an invalid image file.")

                    prediction = model.predict(x=target_image, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(str(result[2] * 100))
                        each_image_details = {}
                        each_image_details["predictions"] = prediction_results
                        each_image_details["percentage_probabilities"] = prediction_probabilities
                        output_array.append(each_image_details)
                    except:
                        raise ValueError("You have set a wrong path to the JSON file")


                elif (self.__modelType == "densenet"):

                    model = self.__model_collection[0]

                    try:
                        from .DenseNet.densenet import DenseNetImageNet121, preprocess_input, decode_predictions
                        from .DenseNet.densenet import DenseNetImageNet121
                        image_to_predict = image.load_img(eachImage,
                                                          grayscale=False, target_size=(224, 224))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict, data_format="channels_last")
                    except:
                        raise ValueError("You have set a path to an invalid image file.")

                    prediction = model.predict(x=image_to_predict, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(str(result[2] * 100))

                        each_image_details = {}
                        each_image_details["predictions"] = prediction_results
                        each_image_details["percentage_probabilities"] = prediction_probabilities
                        output_array.append(each_image_details)
                    except:
                        raise ValueError("You have set a wrong path to the JSON file")


                elif (self.__modelType == "inceptionv3"):

                    model = self.__model_collection[0]

                    try:
                        from imageai.Prediction.InceptionV3.inceptionv3 import InceptionV3
                        from imageai.Prediction.InceptionV3.inceptionv3 import preprocess_input, decode_predictions

                        image_to_predict = image.load_img(eachImage,
                                                          grayscale=False, target_size=(299, 299))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")

                    prediction = model.predict(x=image_to_predict, steps=1)

                    try:
                        predictiondata = decode_predictions(prediction, top=int(result_count_per_image))

                        for results in predictiondata:
                            countdown = 0
                            for result in results:
                                countdown += 1
                                prediction_results.append(str(result[1]))
                                prediction_probabilities.append(str(result[2] * 100))

                        each_image_details = {}
                        each_image_details["predictions"] = prediction_results
                        each_image_details["percentage_probabilities"] = prediction_probabilities
                        output_array.append(each_image_details)
                    except:
                        raise ValueError("You have set a wrong path to the JSON file")

        return output_array



        



