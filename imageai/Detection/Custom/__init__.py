import os
import re
import numpy as np
import json
from imageai.Detection.Custom.voc import parse_voc_annotation
from imageai.Detection.Custom.yolo import create_yolov3_model, dummy_loss
from imageai.Detection.YOLOv3.models import yolo_main
from imageai.Detection.Custom.generator import BatchGenerator
from imageai.Detection.Custom.utils.utils import normalize, evaluate, makedirs
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from imageai.Detection.Custom.callbacks import CustomModelCheckpoint, CustomTensorBoard
from imageai.Detection.Custom.utils.multi_gpu_model import multi_gpu_model
from imageai.Detection.Custom.gen_anchors import generateAnchors
import tensorflow as tf
from keras.models import load_model, Input
from keras.callbacks import TensorBoard
import keras.backend as K
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class DetectionModelTrainer:

    """
    This is the Detection Model training class, which allows you to train object detection models
    on image datasets that are in Pascal VOC annotation format, using the YOLOv3.
    """

    def __init__(self):
        self.__model_type = ""
        self.__training_mode = True

        self.__model_min_input_size = 288
        self.__model_max_input_size = 448
        self.__model_anchors = []
        self.__inference_anchors = []
        self.__json_directory = ""
        self.__model_labels = []
        self.__num_objects = 0
        self.__pre_trained_model = ""

        self.__train_images_folder = ""
        self.__train_annotations_folder = ""
        self.__train_cache_file = ""
        self.__train_times = 8
        self.__train_batch_size = 4
        self.__train_learning_rate = 1e-4
        self.__train_epochs = 100
        self.__train_warmup_epochs = 3
        self.__train_ignore_treshold = 0.5
        self.__train_gpus = "0"
        self.__train_grid_scales = [1, 1, 1]
        self.__train_obj_scale = 5
        self.__train_noobj_scale = 1
        self.__train_xywh_scale = 1
        self.__train_class_scale = 1
        self.__model_directory = ""
        self.__train_weights_name = ""
        self.__train_debug = True
        self.__logs_directory = ""

        self.__validation_images_folder = ""
        self.__validation_annotations_folder = ""
        self.__validation_cache_file = ""
        self.__validation_times = 1

    def setModelTypeAsYOLOv3(self):
        """
        'setModelTypeAsYOLOv3()' is used to set the model type to the YOLOv3 model
        for the training instance object .
        :return:
        """
        self.__model_type = "yolov3"

    def setDataDirectory(self, data_directory):

        """

        'setDataDirectory()' is required to set the path to which the data/dataset to be used for
                 training is kept. The directory can have any name, but it must have 'train' and 'validation'
                 sub-directory. In the 'train' and 'validation' sub-directories, there must be 'images' and 'annotations'
                 sub-directories respectively. The 'images' folder will contain the pictures for the dataset and the
                 'annotations' folder will contain the XML files with details of the annotations for each image in the
                 'images folder'.

                 N.B: Strictly take note that the filenames (without the extension) of the pictures in the 'images folder'
                  must be the same as the filenames (without the extension) of their corresponding annotation XML files in
                  the 'annotations' folder.

                 The structure of the 'train' and 'validation' folder must be as follows:

                >> train    >> images       >> img_1.jpg
                            >> images       >> img_2.jpg
                            >> images       >> img_3.jpg
                            >> annotations  >> img_1.xml
                            >> annotations  >> img_2.xml
                            >> annotations  >> img_3.xml


                >> validation   >> images       >> img_151.jpg
                                >> images       >> img_152.jpg
                                >> images       >> img_153.jpg
                                >> annotations  >> img_151.xml
                                >> annotations  >> img_152.xml
                                >> annotations  >> img_153.xml

        :param data_directory:
        :return:
        """

        self.__train_images_folder = os.path.join(data_directory, "train", "images")
        self.__train_annotations_folder = os.path.join(data_directory, "train", "annotations")
        self.__validation_images_folder = os.path.join(data_directory, "validation", "images")
        self.__validation_annotations_folder = os.path.join(data_directory, "validation", "annotations")

        os.makedirs(os.path.join(data_directory, "cache"), exist_ok=True)
        self.__train_cache_file = os.path.join(data_directory, "cache", "detection_train_data.pkl")
        self.__validation_cache_file = os.path.join(data_directory, "cache", "detection_test_data.pkl")

        os.makedirs(os.path.join(data_directory, "models"), exist_ok=True)

        os.makedirs(os.path.join(data_directory, "json"), exist_ok=True)

        os.makedirs(os.path.join(data_directory, "logs"), exist_ok=True)

        self.__model_directory = os.path.join(data_directory, "models")
        self.__train_weights_name = os.path.join(self.__model_directory, "detection_model-")
        self.__json_directory = os.path.join(data_directory, "json")
        self.__logs_directory = os.path.join(data_directory, "logs")

    def setGpuUsage(self, train_gpus):
        """
        'setGpuUsage' function allows you to set the GPUs to be used while training
        train_gpu can be:
        - an integer, indicating the number of GPUs to use
        - a list of integers, indicating the id of the GPUs to be used
        - a string, indicating the it og the id of the GPUs to be used, separated by commas
        :param train_gpus: gpus where to run
        :return:
        """
        # train_gpus, could be a string separated by comma, or a list of int or the number of GPUs to be used
        if type(train_gpus) == str:
            train_gpus = train_gpus.split(',')
        if type(train_gpus) == int:
            train_gpus = range(train_gpus)
        # let it as a string separated by commas
        self.__train_gpus = ','.join([str(gpu) for gpu in train_gpus])

    def setTrainConfig(self,  object_names_array, batch_size=4, num_experiments=100, train_from_pretrained_model=""):

        """

        'setTrainConfig()' function allows you to set the properties for the training instances. It accepts the following values:

        - object_names_array , this is an array of the names of the different objects in your dataset
        - batch_size (optional),  this is the batch size for the training instance
        - num_experiments (optional),   also known as epochs, it is the number of times the network will train on all the training dataset
        - train_from_pretrained_model (optional), this is used to perform transfer learning by specifying the path to a pre-trained YOLOv3 model

        :param object_names_array:
        :param batch_size:
        :param num_experiments:
        :param train_from_pretrained_model:
        :return:
        """

        self.__model_anchors, self.__inference_anchors = generateAnchors(self.__train_annotations_folder,
                                                                         self.__train_images_folder,
                                                                         self.__train_cache_file, self.__model_labels)

        self.__model_labels = sorted(object_names_array)
        self.__num_objects = len(object_names_array)

        self.__train_batch_size = batch_size
        self.__train_epochs = num_experiments
        self.__pre_trained_model = train_from_pretrained_model

        json_data = dict()
        json_data["labels"] = self.__model_labels
        json_data["anchors"] = self.__inference_anchors

        with open(os.path.join(self.__json_directory, "detection_config.json"), "w+") as json_file:
            json.dump(json_data, json_file, indent=4, separators=(",", " : "),
                      ensure_ascii=True)

        print("Detection configuration saved in ", os.path.join(self.__json_directory, "detection_config.json"))

    def trainModel(self):

        """
        'trainModel()' function starts the actual model training. Once the training starts, the training instance
        creates 3 sub-folders in your dataset folder which are:

        - json,  where the JSON configuration file for using your trained model is stored
        - models, where your trained models are stored once they are generated after each improved experiments
        - cache , where temporary traing configuraton files are stored

        :return:
        """

        train_ints, valid_ints, labels, max_box_per_image = self._create_training_instances(
            self.__train_annotations_folder,
            self.__train_images_folder,
            self.__train_cache_file,
            self.__validation_annotations_folder,
            self.__validation_images_folder,
            self.__validation_cache_file,
            self.__model_labels

        )
        if self.__training_mode:
            print('Training on: \t' + str(labels) + '')
            print("Training with Batch Size: ", self.__train_batch_size)
            print("Number of Training Samples: ", len(train_ints))
            print("Number of Validation Samples: ", len(valid_ints))
            print("Number of Experiments: ", self.__train_epochs)

        ###############################
        #   Create the generators
        ###############################
        train_generator = BatchGenerator(
            instances=train_ints,
            anchors=self.__model_anchors,
            labels=labels,
            downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
            max_box_per_image=max_box_per_image,
            batch_size=self.__train_batch_size,
            min_net_size=self.__model_min_input_size,
            max_net_size=self.__model_max_input_size,
            shuffle=True,
            jitter=0.3,
            norm=normalize
        )

        valid_generator = BatchGenerator(
            instances=valid_ints,
            anchors=self.__model_anchors,
            labels=labels,
            downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
            max_box_per_image=max_box_per_image,
            batch_size=self.__train_batch_size,
            min_net_size=self.__model_min_input_size,
            max_net_size=self.__model_max_input_size,
            shuffle=True,
            jitter=0.0,
            norm=normalize
        )

        ###############################
        #   Create the model
        ###############################
        if os.path.exists(self.__pre_trained_model):
            self.__train_warmup_epochs = 0
        warmup_batches = self.__train_warmup_epochs * (self.__train_times * len(train_generator))

        os.environ['CUDA_VISIBLE_DEVICES'] = self.__train_gpus
        multi_gpu = [int(gpu) for gpu in self.__train_gpus.split(',')]

        train_model, infer_model = self._create_model(
            nb_class=len(labels),
            anchors=self.__model_anchors,
            max_box_per_image=max_box_per_image,
            max_grid=[self.__model_max_input_size, self.__model_max_input_size],
            batch_size=self.__train_batch_size,
            warmup_batches=warmup_batches,
            ignore_thresh=self.__train_ignore_treshold,
            multi_gpu=multi_gpu,
            lr=self.__train_learning_rate,
            grid_scales=self.__train_grid_scales,
            obj_scale=self.__train_obj_scale,
            noobj_scale=self.__train_noobj_scale,
            xywh_scale=self.__train_xywh_scale,
            class_scale=self.__train_class_scale,
        )

        ###############################
        #   Kick off the training
        ###############################
        callbacks = self._create_callbacks(self.__train_weights_name, infer_model)

        train_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator) * self.__train_times,
            validation_data=valid_generator,
            validation_steps=len(valid_generator) * self.__train_times,
            epochs=self.__train_epochs + self.__train_warmup_epochs,
            verbose=1,
            callbacks=callbacks,
            workers=4,
            max_queue_size=8
        )

    def evaluateModel(self, model_path, json_path, batch_size=4, iou_threshold=0.5, object_threshold=0.2, nms_threshold=0.45):
        """

        'evaluateModel()' is used to obtain the mAP metrics for your model(s). It accepts the following values:

        - model_path ( model file or folder), this value can be the part to your model file or the path to the folder containing all your saved model files
        - json_path ,   this is the path the the 'detection_config.json' file saved for the dataset during the training
        - iou_threshold , this value is used to set the desired 'IoU' to obtain the mAP metrics for your model(s)
        - object_threshold , this is used to set your desired minimum 'class score' to obtain the mAP metrics for your model(s)
        - nms_threshold , this is used to set your desired 'Non-maximum suppresion' to obtain the mAP metrics for your model(s)

        :param model_path:
        :param json_path:
        :param batch_size:
        :param iou_threshold:
        :param object_threshold:
        :param nms_threshold:
        :return: list of dictionaries, containing one dict per evaluated model.
            Each dict contains exactly the same metrics that are printed on standard output
        """

        self.__training_mode = False

        with open(json_path, 'r') as json_file:
            detection_model_json = json.load(json_file)

        temp_anchor_array = []
        new_anchor_array = []

        temp_anchor_array.append(detection_model_json["anchors"][2])
        temp_anchor_array.append(detection_model_json["anchors"][1])
        temp_anchor_array.append(detection_model_json["anchors"][0])

        for aa in temp_anchor_array:
            for aaa in aa:
                new_anchor_array.append(aaa)

        self.__model_anchors = new_anchor_array
        self.__model_labels = detection_model_json["labels"]
        self.__num_objects = len(self.__model_labels)

        self.__train_batch_size = batch_size
        self.__train_epochs = 100

        print("Starting Model evaluation....")

        _, valid_ints, labels, max_box_per_image = self._create_training_instances(
            self.__train_annotations_folder,
            self.__train_images_folder,
            self.__train_cache_file,
            self.__validation_annotations_folder,
            self.__validation_images_folder,
            self.__validation_cache_file,
            self.__model_labels

        )

        if len(valid_ints) == 0:
            print('Validation samples were not provided.')
            print('Please, check your validation samples are correctly provided:')
            print('\tAnnotations: {}\n\tImages: {}'.format(self.__validation_annotations_folder,
                                                           self.__validation_images_folder))

        valid_generator = BatchGenerator(
            instances=valid_ints,
            anchors=self.__model_anchors,
            labels=labels,
            downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
            max_box_per_image=max_box_per_image,
            batch_size=self.__train_batch_size,
            min_net_size=self.__model_min_input_size,
            max_net_size=self.__model_max_input_size,
            shuffle=True,
            jitter=0.0,
            norm=normalize
        )

        results = list()

        if os.path.isfile(model_path):
            # model_files must be a list containing the complete path to the files,
            # if a file is given, then the list contains just this file
            model_files = [model_path]
        elif os.path.isdir(model_path):
            # model_files must be a list containing the complete path to the files,
            # if a folder is given, then the list contains the complete path to each file on that folder
            model_files = sorted([os.path.join(model_path, file_name) for file_name in os.listdir(model_path)])
            # sort the files to make sure we're always evaluating them on same order
        else:
            print('model_path must be the path to a .h5 file or a directory. Found {}'.format(model_path))
            return results

        for model_file in model_files:
            if str(model_file).endswith(".h5"):
                try:
                    infer_model = load_model(model_file)

                    ###############################
                    #   Run the evaluation
                    ###############################
                    # compute mAP for all the classes
                    average_precisions = evaluate(infer_model, valid_generator, iou_threshold=iou_threshold,
                                                  obj_thresh=object_threshold, nms_thresh=nms_threshold)

                    result_dict = {
                        'model_file': model_file,
                        'using_iou': iou_threshold,
                        'using_object_threshold': object_threshold,
                        'using_non_maximum_suppression': nms_threshold,
                        'average_precision': dict(),
                        'evaluation_samples': len(valid_ints)
                    }
                    # print the score
                    print("Model File: ", model_file, '\n')
                    print("Evaluation samples: ", len(valid_ints))
                    print("Using IoU: ", iou_threshold)
                    print("Using Object Threshold: ", object_threshold)
                    print("Using Non-Maximum Suppression: ", nms_threshold)

                    for label, average_precision in average_precisions.items():
                        print(labels[label] + ': {:.4f}'.format(average_precision))
                        result_dict['average_precision'][labels[label]] = average_precision

                    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
                    result_dict['map'] = sum(average_precisions.values()) / len(average_precisions)
                    print("===============================")

                    results.append(result_dict)
                except Exception as e:
                    print('skipping the evaluation of {} because following exception occurred: {}'.format(model_file, e))
                    continue
            else:
                print('skipping the evaluation of {} since it\'s not a .h5 file'.format(model_file))

        return results

    def _create_training_instances(self,
            train_annot_folder,
            train_image_folder,
            train_cache,
            valid_annot_folder,
            valid_image_folder,
            valid_cache,
            labels,
    ):

        # parse annotations of the training set
        train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

        # parse annotations of the validation set, if any, otherwise split the training set

        if os.path.exists(valid_annot_folder):
            valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
            print('Evaluating over {} samples taken from {}'.format(len(valid_ints),
                                                                    os.path.dirname(valid_annot_folder)))
        else:

            train_portion = 0.8  # use 80% to train and the remaining 20% to evaluate
            train_valid_split = int(round(train_portion * len(train_ints)))
            np.random.seed(0)
            np.random.shuffle(train_ints)

            valid_ints = train_ints[train_valid_split:]
            train_ints = train_ints[:train_valid_split]
            print('Evaluating over {} samples taken as {:5.2f}% of the training set '
                  'given at {}'.format(len(valid_ints),
                                       (1 - train_portion)*100,
                                       os.path.dirname(train_annot_folder)))

        print('Training over {} samples  given at {}'.format(len(train_ints), os.path.dirname(train_annot_folder)))

        # compare the seen labels with the given labels in config.json
        if len(labels) > 0:
            overlap_labels = set(labels).intersection(set(train_labels.keys()))

            # return None, None, None if some given label is not in the dataset
            if len(overlap_labels) < len(labels):
                if self.__training_mode:
                    print('Some labels have no annotations! Please revise the list of labels in your configuration.')
                return None, None, None, None
        else:
            if self.__training_mode:
                print('No labels are provided. Train on all seen labels.')
                print(train_labels)

            labels = train_labels.keys()

        max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

        return train_ints, valid_ints, sorted(labels), max_box_per_image

    def _create_callbacks(self, saved_weights_name, model_to_save):

        checkpoint = CustomModelCheckpoint(
            model_to_save=model_to_save,
            filepath=saved_weights_name + 'ex-{epoch:03d}--loss-{loss:08.3f}.h5',
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min',
            period=1
        )
        reduce_on_plateau = ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=2,
            verbose=0,
            mode='min',
            epsilon=0.01,
            cooldown=0,
            min_lr=0
        )
        tensor_board = TensorBoard(
            log_dir=self.__logs_directory
        )
        return [checkpoint, reduce_on_plateau, tensor_board]

    def _create_model(
            self,
            nb_class,
            anchors,
            max_box_per_image,
            max_grid, batch_size,
            warmup_batches,
            ignore_thresh,
            multi_gpu,
            lr,
            grid_scales,
            obj_scale,
            noobj_scale,
            xywh_scale,
            class_scale
    ):
        if len(multi_gpu) > 1:
            with tf.device('/cpu:0'):
                template_model, infer_model = create_yolov3_model(
                    nb_class=nb_class,
                    anchors=anchors,
                    max_box_per_image=max_box_per_image,
                    max_grid=max_grid,
                    batch_size=batch_size // len(multi_gpu),
                    warmup_batches=warmup_batches,
                    ignore_thresh=ignore_thresh,
                    grid_scales=grid_scales,
                    obj_scale=obj_scale,
                    noobj_scale=noobj_scale,
                    xywh_scale=xywh_scale,
                    class_scale=class_scale
                )
        else:
            template_model, infer_model = create_yolov3_model(
                nb_class=nb_class,
                anchors=anchors,
                max_box_per_image=max_box_per_image,
                max_grid=max_grid,
                batch_size=batch_size,
                warmup_batches=warmup_batches,
                ignore_thresh=ignore_thresh,
                grid_scales=grid_scales,
                obj_scale=obj_scale,
                noobj_scale=noobj_scale,
                xywh_scale=xywh_scale,
                class_scale=class_scale
            )

            # load the pretrained weight if exists, otherwise load the backend weight only

        if len(self.__pre_trained_model) > 3:
            if self.__training_mode:
                print("Training with transfer learning from pretrained Model")
            template_model.load_weights(self.__pre_trained_model, by_name=True)
        else:
            if self.__training_mode:
                print("Pre-trained Model not provided. Transfer learning not in use.")
                print("Training will start with 3 warmup experiments")

        if len(multi_gpu) > 1:
            train_model = multi_gpu_model(template_model, gpus=multi_gpu)
        else:
            train_model = template_model

        optimizer = Adam(lr=lr, clipnorm=0.001)
        train_model.compile(loss=dummy_loss, optimizer=optimizer)

        return train_model, infer_model


class CustomObjectDetection:

    """
    This is the object detection class for using your custom trained models. It supports your custom trained YOLOv3 model and allows to you to perform object detection in images.
    """

    def __init__(self):
        self.__model_type = ""
        self.__model_path = ""
        self.__model_labels = []
        self.__model_anchors = []
        self.__detection_config_json_path = ""
        self.__input_size = 416
        self.__object_threshold = 0.4
        self.__nms_threshold = 0.4
        self.__model = None
        self.__detection_utils = CustomDetectionUtils(labels=[])

    def setModelTypeAsYOLOv3(self):
        """
        'setModelTypeAsYOLOv3' is used to set your custom detection model as YOLOv3
        :return:
        """
        self.__model_type = "yolov3"

    def setModelPath(self, detection_model_path):
        """
        'setModelPath' is used to specify the filepath to your custom detection model
        :param detection_model_path: path to the .h5 model file.
            Usually is one of those under <data_directory>/models/detection_model-ex-ddd--loss-dddd.ddd.h5
        :return: None
        """
        self.__model_path = detection_model_path

    def setJsonPath(self, configuration_json):
        """
        'setJsonPath' is used to set the filepath to the configuration JSON file for your custom detection model
        :param configuration_json: path to the .json file. Usually it is <data_directory>/json/detection_config.json
        :return: None
        """
        self.__detection_config_json_path = configuration_json

    def loadModel(self):

        """
        'loadModel' is used to load the model into the CustomObjectDetection class
        :return: None
        """

        if self.__model_type == "yolov3":
            detection_model_json = json.load(open(self.__detection_config_json_path))

            self.__model_labels = detection_model_json["labels"]
            self.__model_anchors = detection_model_json["anchors"]

            self.__detection_utils = CustomDetectionUtils(labels=self.__model_labels)

            self.__model = yolo_main(Input(shape=(None, None, 3)), 3, len(self.__model_labels))

            self.__model.load_weights(self.__model_path)

    def detectObjectsFromImage(self, input_image="", output_image_path="", input_type="file", output_type="file",
                               extract_detected_objects=False, minimum_percentage_probability=50, nms_treshold=0.4,
                               display_percentage_probability=True, display_object_name=True, thread_safe=False):

        """

        'detectObjectsFromImage()' function is used to detect objects observable in the given image:
                    * input_image , which can be a filepath or image numpy array in BGR
                    * output_image_path (only if output_type = file) , file path to the output image that will contain the detection boxes and label, if output_type="file"
                    * input_type (optional) , filepath/numpy array of the image. Acceptable values are "file" and "array"
                    * output_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file" and "array"
                    * extract_detected_objects (optional) , option to save each object detected individually as an image and return an array of the objects' image path.
                    * minimum_percentage_probability (optional, 30 by default) , option to set the minimum percentage probability for nominating a detected object for output.
                    * nms_threshold (optional, o.45 by default) , option to set the Non-maximum suppression for the detection
                    * display_percentage_probability (optional, True by default), option to show or hide the percentage probability of each object in the saved/returned detected image
                    * display_display_object_name (optional, True by default), option to show or hide the name of each object in the saved/returned detected image
                    * thread_safe (optional, False by default), enforce the loaded detection model works across all threads if set to true, made possible by forcing all Keras inference to run on the default graph


            The values returned by this function depends on the parameters parsed. The possible values returnable
            are stated as below
            - If extract_detected_objects = False or at its default value and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)

            - If extract_detected_objects = False or at its default value and output_type = 'array' ,
              Then the function will return:

                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)

            - If extract_detected_objects = True and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                2. an array of string paths to the image of each object extracted from the image

            - If extract_detected_objects = True and output_type = 'array', the the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                3. an array of numpy arrays of each object detected in the image

        :param input_image:
        :param output_image_path:
        :param input_type:
        :param output_type:
        :param extract_detected_objects:
        :param minimum_percentage_probability:
        :param nms_treshold:
        :param display_percentage_probability:
        :param display_object_name:
        :param thread_safe:
        :return image_frame:
        :return output_objects_array:
        :return detected_objects_image_array:
        """

        if self.__model is None:
            raise ValueError("You must call the loadModel() function before making object detection.")
        else:
            if output_type == "file":
                # from the image file, lets keep the directory and the filename, but remove its  format
                # if output_image_path is path/to/the/output/image.png
                # then output_image_folder is  path/to/the/output/image
                # let's check if it is in the appropriated format soon to fail early
                output_image_folder, n_subs = re.subn(r'\.(?:jpe?g|png|tif|webp|PPM|PGM)$', '', output_image_path, flags=re.I)
                if n_subs == 0:
                    # if no substitution was done, the given output_image_path is not in a supported format,
                    # raise an error
                    raise ValueError("output_image_path must be the path where to write the image. "
                                     "Therefore it must end as one the following: "
                                     "'.jpg', '.png', '.tif', '.webp', '.PPM', '.PGM'. {} found".format(output_image_path))
                elif extract_detected_objects:
                    # Results must be written as files and need to extract detected objects as images,
                    # let's create a folder to store the object's images
                    objects_dir = output_image_folder + "-objects"

                    os.makedirs(objects_dir, exist_ok=True)

            self.__object_threshold = minimum_percentage_probability / 100
            self.__nms_threshold = nms_treshold

            output_objects_array = []
            detected_objects_image_array = []

            if input_type == "file":
                image = cv2.imread(input_image)
            elif input_type == "array":
                image = input_image
            else:
                raise ValueError("input_type must be 'file' or 'array'. {} found".format(input_type))

            image_frame = image.copy()

            height, width, channels = image.shape

            image = cv2.resize(image, (self.__input_size, self.__input_size))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = image.astype("float32") / 255.

            # expand the image to batch
            image = np.expand_dims(image, 0)

            if self.__model_type == "yolov3":
                if thread_safe == True:
                    with K.get_session().graph.as_default():
                        yolo_results = self.__model.predict(image)
                else:
                    yolo_results = self.__model.predict(image)

                boxes = list()

                for idx, result in enumerate(yolo_results):
                    box_set = self.__detection_utils.decode_netout(result[0], self.__model_anchors[idx],
                                                                   self.__object_threshold, self.__input_size,
                                                                   self.__input_size)
                    boxes += box_set

                self.__detection_utils.correct_yolo_boxes(boxes, height, width, self.__input_size, self.__input_size)

                self.__detection_utils.do_nms(boxes, self.__nms_threshold)

                all_boxes, all_labels, all_scores = self.__detection_utils.get_boxes(boxes, self.__model_labels,
                                                                                     self.__object_threshold)

                for object_box, object_label, object_score in zip(all_boxes, all_labels, all_scores):
                    each_object_details = dict()
                    each_object_details["name"] = object_label
                    each_object_details["percentage_probability"] = object_score

                    if object_box.xmin < 0:
                        object_box.xmin = 0
                    if object_box.ymin < 0:
                        object_box.ymin = 0

                    each_object_details["box_points"] = [object_box.xmin, object_box.ymin, object_box.xmax, object_box.ymax]
                    output_objects_array.append(each_object_details)

                drawn_image = self.__detection_utils.draw_boxes_and_caption(image_frame.copy(), all_boxes, all_labels,
                                                                            all_scores, show_names=display_object_name,
                                                                            show_percentage=display_percentage_probability)

                if extract_detected_objects:

                    for cnt, each_object in enumerate(output_objects_array):

                        splitted_image = image_frame[each_object["box_points"][1]:each_object["box_points"][3],
                                                     each_object["box_points"][0]:each_object["box_points"][2]]
                        if output_type == "file":
                            splitted_image_path = os.path.join(objects_dir, "{}-{:05d}.jpg".format(each_object["name"],
                                                                                                   cnt))

                            cv2.imwrite(splitted_image_path, splitted_image)
                            detected_objects_image_array.append(splitted_image_path)
                        elif output_type == "array":
                            detected_objects_image_array.append(splitted_image.copy())

                if output_type == "file":
                    # we already validated that the output_image_path is a supported by OpenCV one
                    cv2.imwrite(output_image_path, drawn_image)

                if extract_detected_objects:
                    if output_type == "file":
                        return output_objects_array, detected_objects_image_array
                    elif output_type == "array":
                        return drawn_image, output_objects_array, detected_objects_image_array

                else:
                    if output_type == "file":
                        return output_objects_array
                    elif output_type == "array":
                        return drawn_image, output_objects_array


class CustomVideoObjectDetection:


    """

    This is the object detection class for videos and camera live stream inputs using your custom trained detection models. It provides support for your custom YOLOv3 models.

    """

    def __init__(self):
        self.__model_type = ""
        self.__model_path = ""
        self.__model_labels = []
        self.__model_anchors = []
        self.__detection_config_json_path = ""
        self.__model_loaded = False
        self.__input_size = 416
        self.__object_threshold = 0.4
        self.__nms_threshold = 0.4
        self.__detector = []
        self.__detection_utils = CustomDetectionUtils(labels=[])

    def setModelTypeAsYOLOv3(self):

        """
        'setModelTypeAsYOLOv3' is used to set your custom detection model as YOLOv3
        :return:
        """

        self.__model_type = "yolov3"


    def setModelPath(self, detection_model_path):
        """
        'setModelPath' is used to specify the filepath to your custom detection model

        :param detection_model_path:
        :return:
        """
        self.__model_path = detection_model_path


    def setJsonPath(self, configuration_json):
        """
        'setJsonPath' is used to set the filepath to the configuration JSON file for your custom detection model

        :param configuration_json:
        :return:
        """
        self.__detection_config_json_path = configuration_json

    def loadModel(self):
        """
        'loadModel' is used to load the model into the CustomVideoObjectDetection class

        :return:
        """

        if (self.__model_loaded == False):
            if(self.__model_type == "yolov3"):
                detector = CustomObjectDetection()
                detector.setModelTypeAsYOLOv3()
                detector.setModelPath(self.__model_path)
                detector.setJsonPath(self.__detection_config_json_path)
                detector.loadModel()

                self.__detector = detector
                self.__model_loaded = True


    def detectObjectsFromVideo(self, input_file_path="", camera_input=None, output_file_path="", frames_per_second=20,
                               frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False,
                               display_percentage_probability=True, display_object_name=True, save_detected_video=True,
                               per_frame_function=None, per_second_function=None, per_minute_function=None,
                               video_complete_function=None, return_detected_frame=False, detection_timeout = None):




        """

        'detectObjectsFromVideo()' function is used to detect objects observable in the given video path or a camera input:
            * input_file_path , which is the file path to the input video. It is required only if 'camera_input' is not set
            * camera_input , allows you to parse in camera input for live video detections
            * output_file_path , which is the path to the output video. It is required only if 'save_detected_video' is not set to False
            * frames_per_second , which is the number of frames to be used in the output video
            * frame_detection_interval (optional, 1 by default)  , which is the intervals of frames that will be detected.
            * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
            * log_progress (optional) , which states if the progress of the frame processed is to be logged to console
            * display_percentage_probability (optional), can be used to hide or show probability scores on the detected video frames
            * display_object_name (optional), can be used to show or hide object names on the detected video frames
            * save_save_detected_video (optional, True by default), can be set to or not to save the detected video
            * per_frame_function (optional), this parameter allows you to parse in a function you will want to execute after each frame of the video is detected. If this parameter is set to a function, after every video  frame is detected, the function will be executed with the following values parsed into it:
                -- position number of the frame
                -- an array of dictinaries, with each dictinary corresponding to each object detected. Each dictionary contains 'name', 'percentage_probability' and 'box_points'
                -- a dictionary with with keys being the name of each unique objects and value are the number of instances of the object present
                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed as the fourth value into the function

            * per_second_function (optional), this parameter allows you to parse in a function you will want to execute after each second of the video is detected. If this parameter is set to a function, after every second of a video is detected, the function will be executed with the following values parsed into it:
                -- position number of the second
                -- an array of dictionaries whose keys are position number of each frame present in the last second , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame
                -- an array of dictionaries, with each dictionary corresponding to each frame in the past second, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame
                -- a dictionary with its keys being the name of each unique object detected throughout the past second, and the key values are the average number of instances of the object found in all the frames contained in the past second
                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                                                                    as the fifth value into the function

            * per_minute_function (optional), this parameter allows you to parse in a function you will want to execute after each minute of the video is detected. If this parameter is set to a function, after every minute of a video is detected, the function will be executed with the following values parsed into it:
                -- position number of the minute
                -- an array of dictionaries whose keys are position number of each frame present in the last minute , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

                -- an array of dictionaries, with each dictionary corresponding to each frame in the past minute, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

                -- a dictionary with its keys being the name of each unique object detected throughout the past minute, and the key values are the average number of instances of the object found in all the frames contained in the past minute

                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed as the fifth value into the function

            * video_complete_function (optional), this parameter allows you to parse in a function you will want to execute after all of the video frames have been detected. If this parameter is set to a function, after all of frames of a video is detected, the function will be executed with the following values parsed into it:
                -- an array of dictionaries whose keys are position number of each frame present in the entire video , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame
                -- an array of dictionaries, with each dictionary corresponding to each frame in the entire video, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame
                -- a dictionary with its keys being the name of each unique object detected throughout the entire video, and the key values are the average number of instances of the object found in all the frames contained in the entire video

            * return_detected_frame (optionally, False by default), option to obtain the return the last detected video frame into the per_per_frame_function, per_per_second_function or per_per_minute_function

            * detection_timeout (optionally, None by default), option to state the number of seconds of a video that should be detected after which the detection function stop processing the video

        :param input_file_path:
        :param camera_input:
        :param output_file_path:
        :param frames_per_second:
        :param frame_detection_interval:
        :param minimum_percentage_probability:
        :param log_progress:
        :param display_percentage_probability:
        :param display_object_name:
        :param save_detected_video:
        :param per_frame_function:
        :param per_second_function:
        :param per_minute_function:
        :param video_complete_function:
        :param return_detected_frame:
        :param detection_timeout:
        :return output_video_filepath:
        :return counting:
        :return output_objects_array:
        :return output_objects_count:
        :return detected_copy:
        :return this_second_output_object_array:
        :return this_second_counting_array:
        :return this_second_counting:
        :return this_minute_output_object_array:
        :return this_minute_counting_array:
        :return this_minute_counting:
        :return this_video_output_object_array:
        :return this_video_counting_array:
        :return this_video_counting:
        """

        output_frames_dict = {}
        output_frames_count_dict = {}

        input_video = cv2.VideoCapture(input_file_path)
        if (camera_input != None):
            input_video = camera_input

        output_video_filepath = output_file_path + '.avi'

        frame_width = int(input_video.get(3))
        frame_height = int(input_video.get(4))
        output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                       frames_per_second,
                                       (frame_width, frame_height))

        counting = 0
        predicted_numbers = None
        scores = None
        detections = None


        detection_timeout_count = 0
        video_frames_count = 0


        if(self.__model_type == "yolov3"):



            while (input_video.isOpened()):
                ret, frame = input_video.read()

                if (ret == True):

                    detected_frame = frame.copy()

                    video_frames_count += 1
                    if (detection_timeout != None):
                        if ((video_frames_count % frames_per_second) == 0):
                            detection_timeout_count += 1

                        if (detection_timeout_count >= detection_timeout):
                            break

                    output_objects_array = []

                    counting += 1

                    if (log_progress == True):
                        print("Processing Frame : ", str(counting))



                    check_frame_interval = counting % frame_detection_interval

                    if (counting == 1 or check_frame_interval == 0):
                        try:
                            detected_frame, output_objects_array = self.__detector.detectObjectsFromImage(
                                input_image=frame, input_type="array", output_type="array",
                                minimum_percentage_probability=minimum_percentage_probability,
                                display_percentage_probability=display_percentage_probability,
                                display_object_name=display_object_name)
                        except:
                            None


                    output_frames_dict[counting] = output_objects_array

                    output_objects_count = {}
                    for eachItem in output_objects_array:
                        eachItemName = eachItem["name"]
                        try:
                            output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                        except:
                            output_objects_count[eachItemName] = 1

                    output_frames_count_dict[counting] = output_objects_count


                    if (save_detected_video == True):
                        output_video.write(detected_frame)

                    if (counting == 1 or check_frame_interval == 0):
                        if (per_frame_function != None):
                            if (return_detected_frame == True):
                                per_frame_function(counting, output_objects_array, output_objects_count,
                                                   detected_frame)
                            elif (return_detected_frame == False):
                                per_frame_function(counting, output_objects_array, output_objects_count)

                    if (per_second_function != None):
                        if (counting != 1 and (counting % frames_per_second) == 0):

                            this_second_output_object_array = []
                            this_second_counting_array = []
                            this_second_counting = {}

                            for aa in range(counting):
                                if (aa >= (counting - frames_per_second)):
                                    this_second_output_object_array.append(output_frames_dict[aa + 1])
                                    this_second_counting_array.append(output_frames_count_dict[aa + 1])

                            for eachCountingDict in this_second_counting_array:
                                for eachItem in eachCountingDict:
                                    try:
                                        this_second_counting[eachItem] = this_second_counting[eachItem] + \
                                                                         eachCountingDict[eachItem]
                                    except:
                                        this_second_counting[eachItem] = eachCountingDict[eachItem]

                            for eachCountingItem in this_second_counting:
                                this_second_counting[eachCountingItem] = int(this_second_counting[eachCountingItem] / frames_per_second)

                            if (return_detected_frame == True):
                                per_second_function(int(counting / frames_per_second),
                                                    this_second_output_object_array, this_second_counting_array,
                                                    this_second_counting, detected_frame)

                            elif (return_detected_frame == False):
                                per_second_function(int(counting / frames_per_second),
                                                    this_second_output_object_array, this_second_counting_array,
                                                    this_second_counting)

                    if (per_minute_function != None):

                        if (counting != 1 and (counting % (frames_per_second * 60)) == 0):

                            this_minute_output_object_array = []
                            this_minute_counting_array = []
                            this_minute_counting = {}

                            for aa in range(counting):
                                if (aa >= (counting - (frames_per_second * 60))):
                                    this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                    this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                            for eachCountingDict in this_minute_counting_array:
                                for eachItem in eachCountingDict:
                                    try:
                                        this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                                                         eachCountingDict[eachItem]
                                    except:
                                        this_minute_counting[eachItem] = eachCountingDict[eachItem]

                            for eachCountingItem in this_minute_counting:
                                this_minute_counting[eachCountingItem] = int(this_minute_counting[eachCountingItem] / (frames_per_second * 60))

                            if (return_detected_frame == True):
                                per_minute_function(int(counting / (frames_per_second * 60)),
                                                    this_minute_output_object_array, this_minute_counting_array,
                                                    this_minute_counting, detected_frame)

                            elif (return_detected_frame == False):
                                per_minute_function(int(counting / (frames_per_second * 60)),
                                                    this_minute_output_object_array, this_minute_counting_array,
                                                    this_minute_counting)


                else:
                    break

            if (video_complete_function != None):

                this_video_output_object_array = []
                this_video_counting_array = []
                this_video_counting = {}

                for aa in range(counting):
                    this_video_output_object_array.append(output_frames_dict[aa + 1])
                    this_video_counting_array.append(output_frames_count_dict[aa + 1])

                for eachCountingDict in this_video_counting_array:
                    for eachItem in eachCountingDict:
                        try:
                            this_video_counting[eachItem] = this_video_counting[eachItem] + \
                                                            eachCountingDict[eachItem]
                        except:
                            this_video_counting[eachItem] = eachCountingDict[eachItem]

                for eachCountingItem in this_video_counting:
                    this_video_counting[eachCountingItem] = this_video_counting[
                                                                eachCountingItem] / counting

                video_complete_function(this_video_output_object_array, this_video_counting_array,
                                        this_video_counting)

            input_video.release()
            output_video.release()

            if (save_detected_video == True):
                return output_video_filepath


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class CustomDetectionUtils:
    def __init__(self, labels):
        self.__labels = labels
        self.__colors = []

        for i in range(len(labels)):
            color_space_values = np.random.randint(50, 255, size=(3,))
            red, green, blue = color_space_values
            red, green, blue = int(red), int(green), int(blue)
            self.__colors.append([red, green, blue])

    @staticmethod
    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def decode_netout(self, netout, anchors, obj_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = self._sigmoid(netout[..., :2])
        netout[..., 4:] = self._sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # 4th element is objectness score
                    objectness = netout[row, col, b, 4]

                    if objectness <= obj_thresh:
                        continue

                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]
                    x = (col + x) / grid_w  # center position, unit: image width
                    y = (row + y) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
                    # last elements are class probabilities
                    classes = netout[row, col, b, 5:]
                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                    boxes.append(box)

        return boxes

    @staticmethod
    def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
        new_w, new_h = net_w, net_h
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
            y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union = w1 * h1 + w2 * h2 - intersect

        try:
            result = float(intersect) / float(union)
            return result
        except:
            return 0.0

    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return

        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i].classes[c] == 0: continue

                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0

    def get_boxes(self, boxes, labels, thresh):
        v_boxes, v_labels, v_scores = list(), list(), list()
        # enumerate all boxes
        for box in boxes:
            # enumerate all possible labels
            for i in range(len(labels)):
                # check if the threshold for this label is high enough
                if box.classes[i] > thresh:
                    v_boxes.append(box)
                    v_labels.append(labels[i])
                    v_scores.append(box.classes[i] * 100)
                # don't break, many labels may trigger for one box
        return v_boxes, v_labels, v_scores

    def label_color(self, label):
        """ Return a color from a set of predefined colors. Contains 80 colors in total.

        Args
            label: The label to get the color for.

        Returns
            A list of three values representing a RGB color.

            If no color is defined for a certain label, the color green is returned and a warning is printed.
        """
        if label < len(self.__colors):
            return self.__colors[label]
        else:
            return 0, 255, 0

    def draw_boxes_and_caption(self, image_frame, v_boxes, v_labels, v_scores, show_names=False, show_percentage=False):

        for i in range(len(v_boxes)):
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            width, height = x2 - x1, y2 - y1
            class_color = self.label_color(self.__labels.index(v_labels[i]))

            image_frame = cv2.rectangle(image_frame, (x1, y1), (x2, y2), class_color, 2)

            label = ""
            if show_names and show_percentage:
                label = "%s : %.3f" % (v_labels[i], v_scores[i])
            elif show_names:
                label = "%s" % (v_labels[i])
            elif show_percentage:
                label = "%.3f" % (v_scores[i])

            if show_names or show_percentage:
                b = np.array([x1, y1, x2, y2]).astype(int)
                cv2.putText(image_frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 0), 3)
                cv2.putText(image_frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        return image_frame
