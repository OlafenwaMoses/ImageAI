import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib.cbook import deprecated


class ImageClassification:
    """
    This is the image classification class in the ImageAI library. It provides support for 4 different models which are:
    ResNet, MobileNetV2, DenseNet and Inception V3. After instantiating this class, you can set it's properties and
    make image classification using it's pre-defined functions.

    The following functions are required to be called before a classification can be made
    * setModelPath()
    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsMobileNetv2(), setModelTypeAsResNet(), setModelTypeAsDenseNet, setModelTypeAsInceptionV3]
    * loadModel() [This must be called once only before making a classification]

    Once the above functions have been called, you can call the classifyImage() function of the classification instance
    object at anytime to classify an image.
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
        available 4 model types. The model path must correspond to the model type set for the classification instance object.

        :param model_path:
        :return:
        """
        self.modelPath = model_path

    def setModelTypeAsSqueezeNet(self):
        raise ValueError("ImageAI no longer support SqueezeNet. You can use MobileNetV2 instead by downloading the MobileNetV2 model and call the function 'setModelTypeAsMobileNetV2'")

    def setModelTypeAsMobileNetV2(self):
        """
        'setModelTypeAsMobileNetV2()' is used to set the model type to the MobileNetV2 model
        for the classification instance object .
        :return:
        """
        self.__modelType = "mobilenetv2"

    @deprecated(since="2.1.6", message="'.setModelTypeAsResNet()' has been deprecated! Please use 'setModelTypeAsResNet50()' instead.")
    def setModelTypeAsResNet(self):
        return self.setModelTypeAsResNet50()

    def setModelTypeAsResNet50(self):
        """
         'setModelTypeAsResNet50()' is used to set the model type to the ResNet50 model
                for the classification instance object .
        :return:
        """
        self.__modelType = "resnet50"

    @deprecated(since="2.1.6", message="'.setModelTypeAsDenseNet()' has been deprecated! Please use 'setModelTypeAsDenseNet121()' instead.")
    def setModelTypeAsDenseNet(self):
        return self.setModelTypeAsDenseNet121()

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

        if(classification_speed=="normal"):
            self.__input_image_size = 224
        elif(classification_speed=="fast"):
            self.__input_image_size = 160
        elif(classification_speed=="faster"):
            self.__input_image_size = 120
        elif (classification_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):

            if(self.__modelType == "" ):
                raise ValueError("You must set a valid model type before loading the model.")

            elif(self.__modelType == "mobilenetv2"):
                model = tf.keras.applications.MobileNetV2(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes = 1000 )
                model.load_weights(self.modelPath)
                self.__model_collection.append(model)
                self.__modelLoaded = True
                try:
                    None
                except:
                    raise ValueError("An error occured. Ensure your model file is a MobileNetV2 Model and is located in the path {}".format(self.modelPath))

            elif(self.__modelType == "resnet50"):
                try:
                    model = tf.keras.applications.ResNet50(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes = 1000 )
                    model.load_weights(self.modelPath)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except Exception as e:
                    raise ValueError("An error occured. Ensure your model file is a ResNet50 Model and is located in the path {}".format(self.modelPath))

            elif (self.__modelType == "densenet121"):
                try:
                    model = tf.keras.applications.DenseNet121(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes = 1000 )
                    model.load_weights(self.modelPath)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is a DenseNet121 Model and is located in the path {}".format(self.modelPath))

            elif (self.__modelType == "inceptionv3"):
                try:
                    model = tf.keras.applications.InceptionV3(input_shape=(self.__input_image_size, self.__input_image_size, 3), weights=None, classes = 1000 )
                    model.load_weights(self.modelPath)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is in {}".format(self.modelPath))


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
            elif (self.__modelType == "densenet121"):
                image_to_predict = tf.keras.applications.densenet.preprocess_input(image_to_predict)
            elif (self.__modelType == "inceptionv3"):
                image_to_predict = tf.keras.applications.inception_v3.preprocess_input(image_to_predict)

            try:
                model = self.__model_collection[0]
                prediction = model.predict(image_to_predict, steps=1)

                if (self.__modelType == "mobilenetv2"):
                    predictiondata = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=int(result_count))
                elif (self.__modelType == "resnet50"):
                    predictiondata = tf.keras.applications.resnet50.decode_predictions(prediction, top=int(result_count))
                elif (self.__modelType == "inceptionv3"):
                    predictiondata = tf.keras.applications.inception_v3.decode_predictions(prediction, top=int(result_count))
                elif (self.__modelType == "densenet121"):
                    predictiondata = tf.keras.applications.densenet.decode_predictions(prediction, top=int(result_count))

                

                for results in predictiondata:
                    for result in results:
                        classification_results.append(str(result[1]))
                        classification_probabilities.append(result[2] * 100)
            except:
                raise ValueError("An error occured! Try again.")

            return classification_results, classification_probabilities
                

    @deprecated(since="2.1.6", message="'.predictImage()' has been deprecated! Please use 'classifyImage()' instead.")
    def predictImage(self, image_input, result_count=5, input_type="file"):
        
        return self.classifyImage(image_input, result_count, input_type)