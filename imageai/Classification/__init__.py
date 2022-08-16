#!/usr/bin/python3

'''
Init file for image classification
'''

import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib.cbook import deprecated


class ImageClassification:
    '''
    This is the image classification class in the ImageAI library. It provides
    support for 4 different models which are: ResNet, MobileNetV2, DenseNet
    and Inception V3. After instantiating this class, you can set it's
    properties and make image classification using it's pre-defined functions.

    The following functions are required to be called before a classification
    can be made
    * set_model_path()
    * At least of of the following and it must correspond to the model set in
    the set_model_path()
    [setModelTypeAsMobileNetv2(), setModelTypeAsResNet(),
    setModelTypeAsDenseNet, setModelTypeAsInceptionV3]
    * load_model() [This must be called once only before making a
    classification]

    Once the above functions have been called, you can call the
    classify_image() function of the classification instance object at
    anytime to classify an image.
    '''

    def __init__(self):
        self.model_path = ''
        self.__model_type = ''
        self.__model_loaded = False
        self.__model_collection = []

    def set_model_path(self, model_path):
        '''
        'set_model_path()' function is required and is used to set the file
        path to the model adopted from the list of the available 4 model
        types. The model path must correspond to the model type set for the
        classification instance object.

        :param model_path:
        :return:
        '''
        self.model_path = model_path

    def set_model_type(self, model_type):
        '''
        This is a function that simply sets the model type. It was designed
        with future compatibility, making it easy to switch out, update,
        and remove models as they're updated and changed

        Take the model type input and make it lowercase to remove any
        capitalizaton errors then set model type variable
        '''
        match model_type.lower():
            case 'mobilenetv2':
                self.__model_type = 'MobileNetV2'
            case 'resnet50':
                self.__model_type = 'ResNet50'
            case 'densenet121':
                self.__model_type = 'DenseNet121'
            case 'inceptionv3':
                self.__model_type = 'InceptionV3'

    def setModelTypeAsSqueezeNet(self):
        raise ValueError(
            'ImageAI no longer support SqueezeNet. You can use MobileNetV2'
            ' instead by downloading the MobileNetV2 model and call the'
            ' function \'setModelTypeAsMobileNetV2\'')

    @deprecated(since='2.2.0', message='\'.setModelTypeAsMobileNetV2()\' has'
                ' been deprecated! Please use'
                ' \'set_model_type(\'MobileNetV2\')\' instead.')
    def setModelTypeAsMobileNetV2(self):
        self.set_model_type('MobileNetV2')

    @deprecated(since='2.2.0', message='\'.setModelTypeAsResNet50()\' has'
                ' been deprecated! Please use'
                ' \'set_model_type(\'ResNet50\')\' instead.')
    def setModelTypeAsResNet50(self):
        self.set_model_type('ResNet50')

    @deprecated(since='2.2.0', message='\'.setModelTypeAsDenseNet121()\''
                ' has been deprecated! Please use'
                ' \'set_model_type(\'DenseNet121\')\' instead.')
    def setModelTypeAsDenseNet121(self):
        self.set_model_type('DenseNet121')

    @deprecated(since='2.2.0', message='\'.setModelTypeAsInceptionV3()\''
                ' has been deprecated! Please use'
                ' \'set_model_type(\'InceptionV3\')\' instead.')
    def setModelTypeAsInceptionV3(self):
        self.set_model_type('InceptionV3')

    def load_model(self):
        '''
        'load_model()' function is used to load the model structure into the
        program from the file path defined in the set_model_path() function.
        This function receives an optional value which is
        'classification_speed'. The value is used to reduce the time it takes
        to classify an image, down to about 50% of the normal time, with just
        slight changes or drop in classification accuracy, depending on the
        nature of the image.
        * classification_speed (optional); Acceptable values are 'normal',
        'fast', 'faster' and 'fastest'

        :param classification_speed :
        :return:
        '''

        if not self.__model_loaded:

            if self.__model_type == '':
                print('No model type specified, defaulting to ResNet50')
                self.__model_type = 'ResNet50'

            match self.__model_type:
                case 'MobileNetV2':
                    model = tf.keras.applications.mobilenet_v2.MobileNetV2()
                case 'ResNet50':
                    model = tf.keras.applications.resnet50.ResNet50()
                case 'DenseNet121':
                    model = tf.keras.applications.densenet.DenseNet121()
                case 'InceptionV3':
                    model = tf.keras.applications.inception_v3.InceptionV3()
                case _:
                    raise ValueError('You must set a valid model type before'
                                     ' loading the model.')

            self.__model_collection.append(model)
            self.__model_loaded = True

    def classify_image(self, image_input, result_count=5, input_type='file'):
        '''
        'classify_image()' function is used to classify a given image by
        receiving the following arguments:
            * input_type (optional) , the type of input to be parsed.
              Acceptable values are 'file', 'array' and 'stream'
            * image_input , file path/numpy array/image file stream of
              the image.
            * result_count (optional) , the number of classifications to
              be sent which must be whole numbers between 1 and 1000.
              The default is 5.

        This function returns 2 arrays namely 'classification_results' and
        'classification_probabilities'. The 'classification_results' contains
        possible objects classes arranged in descending of their percentage
        probabilities. The 'classification_probabilities' contains the
        percentage probability of each object class. The position of each
        object class in the 'classification_results' array corresponds with
        the positions of the percentage probability in the
        'classification_probabilities' array.


        :param input_type:
        :param image_input:
        :param result_count:
        :return classification_results, classification_probabilities:
        '''
        classification_results = []
        classification_probabilities = []
        if not self.__model_loaded:
            raise ValueError('You must call the load_model() function before'
                             ' making classification.')

        if input_type == 'file':
            image_to_predict = tf.keras.utils\
                .load_img(image_input, target_size=(
                    224, 224))
            image_to_predict = tf.keras.utils.img_to_array(
                image_to_predict)
            image_to_predict = tf.expand_dims(image_to_predict, 0)
        elif input_type == 'array':
            image_input = Image.fromarray(np.uint8(image_input))
            image_input = image_input.resize(
                (224, 224))
            image_input = np.expand_dims(image_input, axis=0)
            image_to_predict = image_input.copy()
            image_to_predict = np.asarray(
                image_to_predict, dtype=np.float64)
        elif input_type == 'stream':
            image_input = Image.open(image_input)
            image_input = image_input.resize(
                (224, 224))
            image_input = np.expand_dims(image_input, axis=0)
            image_to_predict = image_input.copy()
            image_to_predict = np.asarray(
                image_to_predict, dtype=np.float64)

        app = None
        match self.__model_type:
            case 'MobileNetV2':
                app = tf.keras.applications.mobilenet_v2
            case 'ResNet50':
                app = tf.keras.applications.resnet50
            case 'DenseNet121':
                app = tf.keras.applications.densenet
            case 'InceptionV3':
                app = tf.keras.applications.inception_v3
        image_to_predict = app.preprocess_input(image_to_predict)

        model = self.__model_collection[0]
        prediction = model.predict(image_to_predict)

        prediction_data = app.decode_predictions(prediction,
                                                 top=int(result_count))

        for results in prediction_data:
            for result in results:
                classification_results.append(str(result[1]))
                classification_probabilities.append(result[2] * 100)

        return classification_results, classification_probabilities

    @deprecated(since='2.1.6', message='\'.predictImage()\' has been'
                ' deprecated! Please use \'classify_image()\' instead.')
    def predictImage(self, image_input, result_count=5, input_type='file'):
        return self.classify_image(image_input, result_count, input_type)
