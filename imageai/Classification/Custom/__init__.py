#!/usr/bin/python3

'''
Init file for custom classification in the ImageAI package
'''

import os
import warnings
import json
import time

import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib.cbook import deprecated
import matplotlib.pyplot as plt


# pylint: disable=too-many-instance-attributes
class ClassificationModelTrainer:

    '''
        This is the Classification Model training class, that allows you to
        define a deep learning network from the 4 available networks types
        supported by ImageAI which are MobileNetv2, ResNet50, InceptionV3 and
        DenseNet121.
    '''

    def __init__(self):
        self.__model_type = ''
        self.__data_dir = ''
        self.__train_dir = ''
        self.__test_dir = ''
        self.__logs_dir = ''
        self.__num_epochs = 10
        self.__trained_model_dir = ''
        self.__model_class_dir = ''
        self.__initial_learning_rate = 1e-3

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
    # pylint: disable=too-many-arguments

    def set_data_directory(
            self,
            data_directory='',
            train_subdirectory='train',
            test_subdirectory='test',
            models_subdirectory='models',
            json_subdirectory='json',
            logs_subdirectory='logs'):
        '''
        'set_data_directory()'

        - data_directory (optional), will set the path to which the
          data/dataset to be used for training is kept. The directory
          can have any name, but it must have 'train' and 'test'
          sub-directory. In the 'train' and 'test' sub-directories,
          there must be sub-directories with each having it's name
          corresponds to the name/label of the object whose images are
          to be kept. The structure of the 'test' and 'train' folder must
          be as follows:

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

                Defaults to root project folder.

        - single_dataset_directory (optional), when set to True only one
          folder will be looked for for a dataset and when set to False will
          look for two directories (see data_directory). When this is set to
          true the structure of the dataset folder must be as follows:

                >> data >> class1 >> class1_train_images
                        >> class2 >> class2_train_images
                        >> class3 >> class3_train_images
                        >> class4 >> class4_train_images
                        >> class5 >> class5_train_images

                Defaults to False.

        - dataset_directory (optional), subdirectory within 'data_directory'
          where the data set is, is only used if 'single_dataset_directory' is
          set to True. Defaults to 'data'.
        - validation_split (optional), this number must be between 0 and 1 and
          will deicde how the dataset is split up if 'single_dataset_directory'
          is set to True. Ex: when set to 0.2 20% of the dataset will be used
          for testing and 80% will be used for training. Defaults to 0.2.
        - train_subdirectory (optional), subdirectory within 'data_directory'
          where the training set is. Defaults to 'train'.
        - test_subdirectory (optional), subdirectory within 'data_directory'
          where the testing set is. Defaults to 'test'.
        - models_subdirectory (optional), subdirectory within 'data_directory'
          where the output models will be saved. Defaults to 'models'.
        - json_subdirectory (optional), subdirectory within 'data_directory'
          where the model classes json file will be saved. Defaults to 'json'.

        :param data_directory:
        :param train_subdirectory:
        :param test_subdirectory:
        :param models_subdirectory:
        :param json_subdirectory:
        :return:
        '''

        self.__data_dir = data_directory

        self.__train_dir = os.path.join(self.__data_dir, train_subdirectory)
        self.__test_dir = os.path.join(self.__data_dir, test_subdirectory)
        self.__trained_model_dir = os.path.join(
            self.__data_dir, models_subdirectory)
        self.__model_class_dir = os.path.join(
            self.__data_dir, json_subdirectory)
        self.__logs_dir = os.path.join(self.__data_dir, logs_subdirectory)

    def lr_schedule(self, epoch):
        '''
        This function sets the learning rate throughout model training in an
        attempt to increase accuracy
        '''

        # Learning Rate Schedule
        learning_rate = self.__initial_learning_rate
        total_epochs = self.__num_epochs

        check_1 = int(total_epochs * 0.9)
        check_2 = int(total_epochs * 0.8)
        check_3 = int(total_epochs * 0.6)
        check_4 = int(total_epochs * 0.4)

        if epoch > check_1:
            learning_rate *= 1e-4
        elif epoch > check_2:
            learning_rate *= 1e-3
        elif epoch > check_3:
            learning_rate *= 1e-2
        elif epoch > check_4:
            learning_rate *= 1e-1

        return learning_rate

    def build_model(
            self,
            input_shape,
            model_type,
            preprocess_layers,
            num_classes):
        '''
        Function to build a new model for training
        '''

        # Preprocess layers for image scaling
        # If enhance data is set to True add extra preprocessing for better
        # accuracy

        def preprocess_inputs(inputs):
            if preprocess_layers:
                processed = tf.keras.layers.RandomFlip(
                    'horizontal', input_shape=input_shape)(inputs)
                processed = tf.keras.layers.RandomRotation(0.1)(processed)
                processed = tf.keras.layers.RandomZoom(0.1)(processed)
                processed = tf.keras.layers.RandomHeight(0.1)(processed)
                processed = tf.keras.layers.RandomWidth(0.1)(processed)

            return processed

        # Set up inputs and preprocess them
        inputs = tf.keras.Input(shape=input_shape)
        outputs = inputs
        if preprocess_layers:
            print('Using Enhanced Data Generation')
            outputs = preprocess_inputs(inputs)

        if model_type == '':
            print('No model type specified, defaulting to ResNet50')
            model_type = 'ResNet50'

        rescale_value = 1. / 255
        if model_type in ('MobileNetv2', 'InceptionV3'):
            rescale_value = 1. / 127.5
        outputs = tf.keras.layers.Rescaling(rescale_value,
                                            input_shape=input_shape)(outputs)

        # Check for model type and build a model based on it
        match model_type:
            case 'MobileNetV2':
                model_name = 'model_mobilenetv2_ex-{epoch:03d}'\
                    '_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}'
                outputs = tf.keras.applications\
                    .mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                              weights=None,
                                              classes=num_classes,
                                              include_top=False,
                                              pooling='avg')(outputs)
            case 'ResNet50':
                model_name = 'model_resnet50_ex-{epoch:03d}'\
                    '_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}'
                outputs = tf.keras.applications\
                    .resnet50.ResNet50(input_shape=input_shape,
                                       weights=None,
                                       classes=num_classes,
                                       include_top=False,
                                       pooling='avg')(outputs)
            case 'DenseNet121':
                model_name = 'model_densenet121_ex-{epoch:03d}'\
                    '_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}'
                outputs = tf.keras.applications\
                    .densenet.DenseNet121(input_shape=input_shape,
                                          weights=None,
                                          classes=num_classes,
                                          include_top=False,
                                          pooling='avg')(outputs)
            case 'InceptionV3':
                model_name = 'model_inceptionv3_ex-{epoch:03d}'\
                    '_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}'
                outputs = tf.keras.applications\
                    .inception_v3.InceptionV3(input_shape=input_shape,
                                              weights=None,
                                              classes=num_classes,
                                              include_top=False,
                                              pooling='avg')(outputs)

        # Build final output and create model
        outputs = tf.keras.layers.Dense(
            num_classes, activation='softmax', use_bias=True)(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model to get ready for training
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(
        ), optimizer='adam', metrics=['accuracy'])

        return model, model_name

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def train_model(
            self,
            num_experiments=200,
            preprocess_layers=False,
            batch_size=32,
            initial_learning_rate=1e-3,
            show_network_summary=False,
            training_image_size=224,
            transfer_from_model=None,
            continue_from_model=None,
            show_training_graph=False):
        '''
        'train_model()' function starts the model actual training. It accepts
        the following values:

        - num_experiments (optional), also known as epochs, it is the number
          of times the network will train on all the training dataset
        - preprocess_layers (optional) , this is used to modify the dataset
          and create more instance of the training set to enhance the training
          result
        - batch_size (optional) , due to memory constraints, the network
          trains on a batch at once, until all the training set is exhausted.
          The value is set to 32 by default, but can be increased or decreased
          depending on the meormory of the compute used for training. The
          batch_size is conventionally set to 16, 32, 64, 128.
        - initial_learning_rate (optional) , this value is used to adjust
          the weights generated in the network. You rae advised to keep this
          value as it is if you don't have deep understanding of this concept.
        - show_network_summary (optional) , this value is used to show the
          structure of the network should you desire to see it. It is set to
          False by default
        - training_image_size (optional) , this value is used to define the
          image size on which the model will be trained. The value is 224 by
          default and is kept at a minimum of 100.
        - transfer_from_model (optional) , this is used to set the path to a
          model file trained on another dataset. It is primarily used to
          perform tramsfer learning.
        - show_training_graph (optional), when set to True a graph plotting
          accuracy with validation accuracy as well as loss with validation
          loss at the end of training. It is set to False by default.
        - keep_only_best (optional), when set to True all models saves (full
          or just weights) other than the best will be deleted. Set to True by
          default.


        :param num_experiments:
        :param preprocess_layers:
        :param batch_size:
        :param initial_learning_rate:
        :param show_network_summary:
        :param training_image_size:
        :param transfer_from_model:
        :param save_full_model:
        :return:
        '''

        self.__num_epochs = num_experiments
        self.__initial_learning_rate = initial_learning_rate
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            self.lr_schedule)

        if training_image_size < 100:
            warnings.warn(f'''The specified training_image_size
                          {training_image_size} is less than 100. Hence the
                          training_image_size will default to 100.''')
            training_image_size = 100

        input_shape = (training_image_size, training_image_size, 3)

        # Set up training and validation datasets with caching and prefetching
        # for improved training performance
        autotune = tf.data.AUTOTUNE

        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.__train_dir,
            seed=123,
            image_size=(training_image_size, training_image_size),
            batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.__test_dir,
            seed=123,
            image_size=(training_image_size, training_image_size),
            batch_size=batch_size)

        class_names = train_ds.class_names
        num_classes = len(class_names)

        train_ds = train_ds.cache().shuffle(1000)
        train_ds = train_ds.prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)

        if continue_from_model is None:
            model, model_name = self.build_model(
                input_shape, self.__model_type, preprocess_layers, num_classes)

            # If continuing or transfering a model set load those weights
            if transfer_from_model is not None:
                model.load_weights(transfer_from_model)
        else:
            model = tf.keras.models.load_model(continue_from_model)
            model_names = ['mobilenet_v2', 'resnet50',
                           'densenet121', 'inception_v3']
            model_name = ''
            for layer in model.layers:
                if layer.name in model_names:
                    if layer.name == model_names[0]:
                        model_name = 'model_mobilenetv2_ex-{epoch:03d}'\
                            '_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}'
                    elif layer.name == model_names[1]:
                        model_name = 'model_resnet50_ex-{epoch:03d}'\
                            '_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}'
                    elif layer.name == model_names[2]:
                        model_name = 'model_densenet121_ex-{epoch:03d}'\
                            '_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}'
                    elif layer.name == model_names[3]:
                        model_name = 'model_inceptionv3_ex-{epoch:03d}'\
                            '_acc-{accuracy:.3f}_vacc{val_accuracy:.3f}'
                    break

        # Print model summary if set and print if using a previous model
        if show_network_summary:
            if transfer_from_model is not None:
                print('Training using weights from a previous model')
            model.summary()

        # Set templates for model name, model path, log name, and log path
        model_path = os.path.join(self.__trained_model_dir, model_name)
        formatted_time = time.strftime('%Y-%m-%d-%H-%M-%S')
        log_name = f'{self.__model_type}_lr-'\
            f'{initial_learning_rate}_{formatted_time}'
        log_path = os.path.join(self.__logs_dir, log_name)

        # Check if data directories exist and if they don't create them
        directories = [self.__trained_model_dir, self.__model_class_dir,
                       self.__logs_dir, log_path]
        for dirs in directories:
            if not os.path.isdir(dirs):
                os.makedirs(dirs)

        # Create a checkpoint callback to save models as they're completed
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                        monitor='accuracy',
                                                        verbose=1,
                                                        save_best_only=True)

        # Create a model class json file
        class_json = {}
        for i, class_name in enumerate(class_names):
            class_json[str(i)] = class_name
        with open(os.path.join(self.__model_class_dir, 'model_class.json'),
                  'w+', encoding='UTF-8') as json_file:
            json.dump(class_json, json_file, indent=4, separators=(',', ' : '),
                      ensure_ascii=True)
        print('JSON Mapping for the model classes saved to ',
              os.path.join(self.__model_class_dir, 'model_class.json'))

        # Print number of experiments that will be run
        print('Number of experiments (Epochs) : ', self.__num_epochs)

        # Train the model
        history = model.fit(train_ds, epochs=self.__num_epochs,
                            validation_data=val_ds,
                            callbacks=[checkpoint, lr_scheduler])

        if show_training_graph:
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(self.__num_epochs)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

    def setModelTypeAsSqueezeNet(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        raise ValueError(
            'ImageAI no longer support SqueezeNet. You can use MobileNetV2'
            ' instead by downloading the MobileNetV2 model and call the'
            ' function \'setModelTypeAsMobileNetV2\'')

    @ deprecated(since='2.1.6', message='\'.setModelTypeAsResNet()\' has been'
                 ' deprecated! Please use'
                 ' \'set_model_type(\'ResNet50\')\' instead.')
    def setModelTypeAsResNet(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.setModelTypeAsResNet50()

    @ deprecated(since='2.1.6', message='\'.setModelTypeAsDenseNet()\' has'
                 ' been deprecated! Please use'
                 ' \'set_model_type(\'DenseNet121\')\' instead.')
    def setModelTypeAsDenseNet(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('DenseNet121')

    @ deprecated(since='2.2.0', message='\'.setModelTypeAsMobileNetV2()\' has'
                 ' been deprecated! Please use'
                 ' \'set_model_type(\'MobileNetV2\')\' instead.')
    def setModelTypeAsMobileNetV2(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('MobileNetV2')

    @ deprecated(since='2.2.0', message='\'.setModelTypeAsResNet50()\' has'
                 ' been deprecated! Please use'
                 ' \'set_model_type(\'ResNet50\')\' instead.')
    def setModelTypeAsResNet50(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('ResNet50')

    @ deprecated(since='2.2.0', message='\'.setModelTypeAsDenseNet121()\' has'
                 ' been deprecated! Please use'
                 ' \'set_model_type(\'DenseNet121\')\' instead.')
    def setModelTypeAsDenseNet121(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('DenseNet121')

    @ deprecated(since='2.2.0', message='\'.setModelTypeAsInceptionV3()\' has'
                 ' been deprecated! Please use'
                 ' \'set_model_type(\'InceptionV3\')\' instead.')
    def setModelTypeAsInceptionV3(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('InceptionV3')


class CustomImageClassification:

    '''
    This is the image classification class for custom models trained with the
    'ClassificationModelTrainer' class. It provides support for 4 different
    models which are: ResNet50, MobileNetV2, DenseNet121 and Inception V3.
    After instantiating this class, you can set it's properties and make image
    classification using it's pre-defined functions.

    The following functions are required to be called before a classification
    can be made
    * set_model_path() , path to your custom model
    * set_json_path , , path to your custom model's corresponding JSON file
    * At least of of the following and it must correspond to the model set in
      the set_model_path() [setModelTypeAsMobileNetV2(),
      setModelTypeAsResNet50(), setModelTypeAsDenseNet121,
      setModelTypeAsInceptionV3]
    * load_trained_model() [This must be called once only before making a
      classification]

    Once the above functions have been called, you can call the
    classify_image() function of the classification instance object
    at anytime to predict an image.
    '''

    def __init__(self):
        self.__model_type = ''
        self.model_path = ''
        self.json_path = ''
        self.num_objects = 10
        self.__model_classes = {}
        self.__model_loaded = False
        self.__model_collection = []
        self.__input_image_size = 224

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

    def set_json_path(self, model_json):
        '''
        'set_json_path()' function is not required unless model json is not
        named 'model_class.json' in the 'json' directory of project root
        folder.

        :param model_path:
        :return:
        '''

        self.json_path = model_json

    def set_model_type(self, model_type):
        '''
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

    def get_json(self):
        '''
        If json path is specified open it, otherwise check the default
        location, if json path is not specified nor in the default
        location raise ValueError
        '''

        if self.json_path != '':
            with open(self.json_path, 'r', encoding='UTF-8') as json_file:
                self.__model_classes = json.load(json_file)
        else:
            try:
                with open('json/model_class.json', 'r', encoding='UTF-8')\
                        as json_file:
                    self.__model_classes = json.load(json_file)
            except Exception as exc:
                raise ValueError(
                    'There was an error when loading the model class json\
                     file. Make sure the file location is\
                     \'json\\model_class.json\' or set the json path with\
                     set_json_path()') from exc

    def load_trained_model(self, classification_speed='normal'):
        '''
        'load_trained_model()' function is used to load the model structure
        into the program from the file path defined in the set_model_path()
        function. This function receives an optional value which is
        'classification_speed'. The value is used to reduce the time it takes
        to classify an image, down to about 50% of the normal time, with just
        slight changes or drop in classification accuracy, depending on the
        nature of the image. * classification_speed (optional); Acceptable
        values are 'normal', 'fast', 'faster' and 'fastest'

        :param classification_speed:
        :return:
        '''

        # Call get_json to load all model classes
        self.get_json()

        # Adjust the size that the input image will be rescaled to
        # Smaller numbers mean smaller picture which means quicker
        # analysis and prediction
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

        input_shape = (self.__input_image_size, self.__input_image_size, 3)

        # Create model inputs then rescale inputs to image size which is
        # used by all models
        inputs = tf.keras.Input(
            shape=(self.__input_image_size, self.__input_image_size, 3))
        rescaled = tf.keras.layers.Rescaling(
            1. / 255, input_shape=input_shape)(inputs)

        # Only load model if a model hasn't already been loaded
        if not self.__model_loaded:

            # Get number of objects from model class list length
            num_classes = len(self.__model_classes)
            try:

                # Set outputs based on model type
                if self.__model_type == '':
                    print('No model type specified, defaulting to ResNet50')
                    self.__model_type = 'ResNet50'

                match self.__model_type:
                    case 'MobileNetV2':
                        outputs = tf.keras.applications\
                            .mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                      weights=None,
                                                      classes=num_classes,
                                                      include_top=False,
                                                      pooling='avg')(rescaled)
                    case 'ResNet50':
                        outputs = tf.keras.applications\
                            .resnet50.ResNet50(input_shape=input_shape,
                                               weights=None,
                                               classes=num_classes,
                                               include_top=False,
                                               pooling='avg')(rescaled)
                    case 'DenseNet121':
                        outputs = tf.keras.applications\
                            .densenet.DenseNet121(input_shape=input_shape,
                                                  weights=None,
                                                  classes=num_classes,
                                                  include_top=True,
                                                  pooling='avg')(rescaled)
                    case 'InceptionV3':
                        outputs = tf.keras.applications\
                            .inception_v3.InceptionV3(input_shape=input_shape,
                                                      weights=None,
                                                      classes=num_classes,
                                                      include_top=False,
                                                      pooling='avg')(rescaled)
                    case _:
                        raise ValueError(
                            'You must set a valid model type before'
                            ' loading the model.')

                # Create model, then load weights into the model
                outputs = tf.keras.layers.Dense(
                    num_classes, activation='softmax', use_bias=True)(outputs)
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.load_weights(self.model_path)

                self.__model_collection.append(model)
                self.__model_loaded = True

            except Exception as exc:
                raise ValueError(
                    f'An error occured. Ensure your model file is a'
                    f' {self.__model_type} Model and is located in the'
                    f' path {self.model_path}') from exc

    def lead_full_model(self, classification_speed='normal'):
        '''
        'lead_full_model()' function is used to load the model structure into
        the program from the file path defined in the set_model_path()
        function. As opposed to the 'load_trained_model()' function, you don't
        need to specify the model type. This means you can load any Keras
        model trained with or without ImageAI and perform image prediction.
        - prediction_speed (optional), Acceptable values are 'normal', 'fast',
          'faster' and 'fastest'

        :param prediction_speed:
        :return:
        '''

        self.get_json()

        # Adjust the size that the input image will be rescaled to smaller
        # numbers mean smaller picture which means quicker analysis and
        # prediction, the trade-off to faster speeds is prediction accuracy

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
        if not self.__model_loaded:

            model = tf.keras.models.load_model(filepath=self.model_path)
            self.__model_collection.append(model)
            self.__model_loaded = True
            self.__model_type = 'full'

    # pylint: disable=too-many-locals
    def classify_image(
            self,
            image_input,
            result_count=3,
            input_type='file',
            show_output=True):
        '''
        'classify_image()' function is used to classify a given image by
        receiving the following arguments:
        * input_type (optional) , the type of input to be parsed. Acceptable
          values are 'file', 'array' and 'stream'
        * image_input , file path/numpy array/image file stream of the image.
        * result_count (optional) , the number of classifications to be sent
          which must be whole numbers between 1 and 1000. The default is 5.

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
            raise ValueError(
                'You must call the load_trained_model() function before making'
                ' classification.')

        if input_type == 'file':
            try:
                image_to_predict = tf.keras.utils\
                    .load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                image_to_predict = tf.keras.utils.img_to_array(
                    image_to_predict)
                image_to_predict = tf.expand_dims(image_to_predict, 0)
            except Exception as exc:
                raise ValueError(
                    'You have set a path to an invalid image file.')\
                    from exc
        elif input_type == 'array':
            try:
                image_to_predict = Image.fromarray(image_input)
                image_to_predict = image_to_predict.resize(
                    (self.__input_image_size, self.__input_image_size))
                image_to_predict = tf.keras.utils.img_to_array(
                    image_to_predict)
                image_to_predict = tf.expand_dims(image_to_predict, 0)
            except Exception as exc:
                raise ValueError(
                    'You have parsed in a wrong array for the image')\
                    from exc
        elif input_type == 'stream':
            try:
                image_to_predict = Image.open(image_input)
                image_to_predict = image_to_predict.resize(
                    (self.__input_image_size, self.__input_image_size))
                image_to_predict = tf.expand_dims(image_to_predict, axis=0)
                image_to_predict = image_to_predict.copy()
                image_to_predict = np.asarray(
                    image_to_predict, dtype=np.float64)

            except Exception as exc:
                raise ValueError(
                    'You have parsed in a wrong stream for the image')\
                    from exc
        try:
            model = self.__model_collection[0]
            prediction = model.predict(
                image_to_predict, verbose=1 if show_output is True else 0)

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

        except Exception as exc:
            raise ValueError('Error. Ensure your input image is valid')\
                from exc

        return classification_results, classification_probabilities

    def setModelTypeAsSqueezeNet(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        raise ValueError('''ImageAI no longer support SqueezeNet. You can use
            MobileNetV2 instead by downloading the MobileNetV2 model and
            call the function 'setModelTypeAsMobileNetV2\'''')

    @ deprecated(since='2.1.6', message='\'.predictImage()\' has been'
                 ' deprecated! Please use \'classify_image()\' instead.')
    # pylint: disable=invalid-name
    def predictImage(self, image_input, result_count=3, input_type='file'):
        '''Function set in place to redirect users due to depreceation'''
        return self.classify_image(image_input, result_count, input_type)

    @ deprecated(since='2.1.6', message='\'.setModelTypeAsResNet()\' has'
                 ' been deprecated! Please use'
                 ' \'set_model_type(\'ResNet50\')\' instead.')
    def setModelTypeAsResNet(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.setModelTypeAsResNet50()

    @ deprecated(since='2.1.6', message='\'.setModelTypeAsDenseNet()\' has'
                 ' been deprecated! Please use'
                 ' \'set_model_type(\'DenseNet121\')\' instead.')
    def setModelTypeAsDenseNet(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('DenseNet121')

    @ deprecated(since='2.2.0', message='\'.setModelTypeAsMobileNetV2()\''
                 ' has been deprecated! Please use'
                 ' \'set_model_type(\'MobileNetV2\')\' instead.')
    def setModelTypeAsMobileNetV2(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('MobileNetV2')

    @ deprecated(since='2.2.0', message='\'.setModelTypeAsResNet50()\' has'
                 ' been deprecated! Please use'
                 ' \'set_model_type(\'ResNet50\')\' instead.')
    def setModelTypeAsResNet50(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('ResNet50')

    @ deprecated(since='2.2.0', message='\'.setModelTypeAsDenseNet121()\''
                 ' has been deprecated! Please use'
                 ' \'set_model_type(\'DenseNet121\')\' instead.')
    def setModelTypeAsDenseNet121(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('DenseNet121')

    @ deprecated(since='2.2.0', message='\'.setModelTypeAsInceptionV3()\''
                 ' has been deprecated! Please use'
                 ' \'set_model_type(\'InceptionV3\')\' instead.')
    def setModelTypeAsInceptionV3(self):  # pylint: disable=invalid-name
        '''Function set in place to redirect users due to depreceation'''
        return self.set_model_type('InceptionV3')
