import cv2

from imageai.Detection.keras_retinanet.models.resnet import resnet50_retinanet
from imageai.Detection.keras_retinanet.utils.image import read_image_bgr, read_image_array, read_image_stream, \
    preprocess_image, resize_image
from imageai.Detection.keras_retinanet.utils.visualization import draw_box, draw_caption
from imageai.Detection.keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import numpy as np
import tensorflow as tf
import os
from keras import backend as K
from keras.layers import Input
from PIL import Image
import colorsys

from imageai.Detection.YOLOv3.models import yolo_main, tiny_yolo_main
from imageai.Detection.YOLOv3.utils import letterbox_image, yolo_eval

import traceback

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class ObjectDetection:
    """
                    This is the object detection class for images in the ImageAI library. It provides support for RetinaNet
                     , YOLOv3 and TinyYOLOv3 object detection networks . After instantiating this class, you can set it's properties and
                     make object detections using it's pre-defined functions.

                     The following functions are required to be called before object detection can be made
                     * setModelPath()
                     * At least of of the following and it must correspond to the model set in the setModelPath()
                      [setModelTypeAsRetinaNet(), setModelTypeAsYOLOv3(), setModelTypeAsTinyYOLOv3()]
                     * loadModel() [This must be called once only before performing object detection]

                     Once the above functions have been called, you can call the detectObjectsFromImage() function of
                     the object detection instance object at anytime to obtain observable objects in any image.
    """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.__modelPathAdded = False
        self.__modelLoaded = False
        self.__model_collection = []

        # Instance variables for RetinaNet Model
        self.__input_image_min = 1333
        self.__input_image_max = 800

        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair drier',
                                 79: 'toothbrush'}

        # Unique instance variables for YOLOv3 and TinyYOLOv3 model
        self.__yolo_iou = 0.45
        self.__yolo_score = 0.1
        self.__yolo_anchors = np.array(
            [[10., 13.], [16., 30.], [33., 23.], [30., 61.], [62., 45.], [59., 119.], [116., 90.], [156., 198.],
             [373., 326.]])
        self.__yolo_model_image_size = (416, 416)
        self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = "", "", ""
        self.sess = K.get_session()

        # Unique instance variables for TinyYOLOv3.
        self.__tiny_yolo_anchors = np.array(
            [[10., 14.], [23., 27.], [37., 58.], [81., 82.], [135., 169.], [344., 319.]])

    def setModelTypeAsRetinaNet(self):
        """
        'setModelTypeAsRetinaNet()' is used to set the model type to the RetinaNet model
        for the video object detection instance instance object .
        :return:
        """
        self.__modelType = "retinanet"

    def setModelTypeAsYOLOv3(self):
        """
                'setModelTypeAsYOLOv3()' is used to set the model type to the YOLOv3 model
                for the video object detection instance instance object .
                :return:
                """

        self.__modelType = "yolov3"

    def setModelTypeAsTinyYOLOv3(self):
        """
                        'setModelTypeAsTinyYOLOv3()' is used to set the model type to the TinyYOLOv3 model
                        for the video object detection instance instance object .
                        :return:
                        """

        self.__modelType = "tinyyolov3"

    def setModelPath(self, model_path):
        """
         'setModelPath()' function is required and is used to set the file path to a RetinaNet
          object detection model trained on the COCO dataset.
          :param model_path:
          :return:
        """

        if (self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True

    def loadModel(self, detection_speed="normal"):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function. This function receives an optional value which is "detection_speed".
                The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
                 with just slight reduction in the number of objects detected.


                * prediction_speed (optional); Acceptable values are "normal", "fast", "faster", "fastest" and "flash"

                :param detection_speed:
                :return:
        """

        if (self.__modelType == "retinanet"):
            if (detection_speed == "normal"):
                self.__input_image_min = 800
                self.__input_image_max = 1333
            elif (detection_speed == "fast"):
                self.__input_image_min = 400
                self.__input_image_max = 700
            elif (detection_speed == "faster"):
                self.__input_image_min = 300
                self.__input_image_max = 500
            elif (detection_speed == "fastest"):
                self.__input_image_min = 200
                self.__input_image_max = 350
            elif (detection_speed == "flash"):
                self.__input_image_min = 100
                self.__input_image_max = 250
        elif (self.__modelType == "yolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (416, 416)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (208, 208)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (128, 128)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (96, 96)

        elif (self.__modelType == "tinyyolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (832, 832)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (576, 576)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (416, 416)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (272, 272)

        if (self.__modelLoaded == False):
            if (self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif (self.__modelType == "retinanet"):
                model = resnet50_retinanet(num_classes=80)
                model.load_weights(self.modelPath)
                self.__model_collection.append(model)
                self.__modelLoaded = True
            elif (self.__modelType == "yolov3"):
                model = yolo_main(Input(shape=(None, None, 3)), len(self.__yolo_anchors) // 3,
                                  len(self.numbers_to_names))
                model.load_weights(self.modelPath)

                hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                              for x in range(len(self.numbers_to_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)
                np.random.shuffle(self.colors)
                np.random.seed(None)

                self.__yolo_input_image_shape = K.placeholder(shape=(2,))
                self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = yolo_eval(model.output,
                                                                                       self.__yolo_anchors,
                                                                                       len(self.numbers_to_names),
                                                                                       self.__yolo_input_image_shape,
                                                                                       score_threshold=self.__yolo_score,
                                                                                       iou_threshold=self.__yolo_iou)

                self.__model_collection.append(model)
                self.__modelLoaded = True

            elif (self.__modelType == "tinyyolov3"):
                model = tiny_yolo_main(Input(shape=(None, None, 3)), len(self.__tiny_yolo_anchors) // 2,
                                       len(self.numbers_to_names))
                model.load_weights(self.modelPath)

                hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                              for x in range(len(self.numbers_to_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)
                np.random.shuffle(self.colors)
                np.random.seed(None)

                self.__yolo_input_image_shape = K.placeholder(shape=(2,))
                self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = yolo_eval(model.output,
                                                                                       self.__tiny_yolo_anchors,
                                                                                       len(self.numbers_to_names),
                                                                                       self.__yolo_input_image_shape,
                                                                                       score_threshold=self.__yolo_score,
                                                                                       iou_threshold=self.__yolo_iou)

                self.__model_collection.append(model)
                self.__modelLoaded = True

    def detectObjectsFromImage(self, input_image="", output_image_path="", input_type="file", output_type="file",
                               extract_detected_objects=False, minimum_percentage_probability=50,
                               display_percentage_probability=True, display_object_name=True):
        """
            'detectObjectsFromImage()' function is used to detect objects observable in the given image path:
                    * input_image , which can be file to path, image numpy array or image file stream
                    * output_image_path (only if output_type = file) , file path to the output image that will contain the detection boxes and label, if output_type="file"
                    * input_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file", "array" and "stream"
                    * output_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file" and "array"
                    * extract_detected_objects (optional) , option to save each object detected individually as an image and return an array of the objects' image path.
                    * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
                    * display_percentage_probability (optional, True by default), option to show or hide the percentage probability of each object in the saved/returned detected image
                    * display_display_object_name (optional, True by default), option to show or hide the name of each object in the saved/returned detected image


            The values returned by this function depends on the parameters parsed. The possible values returnable
            are stated as below
            - If extract_detected_objects = False or at its default value and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (tuple of x1,y1,x2 and y2 coordinates)

            - If extract_detected_objects = False or at its default value and output_type = 'array' ,
              Then the function will return:

                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (tuple of x1,y1,x2 and y2 coordinates)

            - If extract_detected_objects = True and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (tuple of x1,y1,x2 and y2 coordinates)
                2. an array of string paths to the image of each object extracted from the image

            - If extract_detected_objects = True and output_type = 'array', the the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (tuple of x1,y1,x2 and y2 coordinates)
                3. an array of numpy arrays of each object detected in the image


            :param input_image:
            :param output_image_path:
            :param input_type:
            :param output_type:
            :param extract_detected_objects:
            :param minimum_percentage_probability:
            :param display_percentage_probability:
            :param display_object_name
            :return output_objects_array:
            :return detected_copy:
            :return detected_detected_objects_image_array:
        """

        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making object detection.")
        elif (self.__modelLoaded == True):
            try:
                if (self.__modelType == "retinanet"):
                    output_objects_array = []
                    detected_objects_image_array = []

                    if (input_type == "file"):
                        image = read_image_bgr(input_image)
                    elif (input_type == "array"):
                        image = read_image_array(input_image)
                    elif (input_type == "stream"):
                        image = read_image_stream(input_image)

                    detected_copy = image.copy()
                    detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                    detected_copy2 = image.copy()
                    detected_copy2 = cv2.cvtColor(detected_copy2, cv2.COLOR_BGR2RGB)

                    image = preprocess_image(image)
                    image, scale = resize_image(image, min_side=self.__input_image_min, max_side=self.__input_image_max)

                    model = self.__model_collection[0]
                    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
                    predicted_numbers = np.argmax(detections[0, :, 4:], axis=1)
                    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_numbers]

                    detections[0, :, :4] /= scale

                    min_probability = minimum_percentage_probability / 100
                    counting = 0

                    for index, (label, score), in enumerate(zip(predicted_numbers, scores)):
                        if score < min_probability:
                            continue

                        counting += 1

                        objects_dir = output_image_path + "-objects"
                        if (extract_detected_objects == True and output_type == "file"):
                            if (os.path.exists(objects_dir) == False):
                                os.mkdir(objects_dir)

                        color = label_color(label)

                        detection_details = detections[0, index, :4].astype(int)
                        draw_box(detected_copy, detection_details, color=color)

                        if (display_object_name == True and display_percentage_probability == True):
                            caption = "{} {:.3f}".format(self.numbers_to_names[label], (score * 100))
                            draw_caption(detected_copy, detection_details, caption)
                        elif (display_object_name == True):
                            caption = "{} ".format(self.numbers_to_names[label])
                            draw_caption(detected_copy, detection_details, caption)
                        elif (display_percentage_probability == True):
                            caption = " {:.3f}".format((score * 100))
                            draw_caption(detected_copy, detection_details, caption)

                        each_object_details = {}
                        each_object_details["name"] = self.numbers_to_names[label]
                        each_object_details["percentage_probability"] = score * 100
                        each_object_details["box_points"] = detection_details

                        output_objects_array.append(each_object_details)

                        if (extract_detected_objects == True):
                            splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
                                            detection_details[0]:detection_details[2]]
                            if (output_type == "file"):
                                splitted_image_path = os.path.join(objects_dir,
                                                                   self.numbers_to_names[label] + "-" + str(
                                                                       counting) + ".jpg")
                                pltimage.imsave(splitted_image_path, splitted_copy)
                                detected_objects_image_array.append(splitted_image_path)
                            elif (output_type == "array"):
                                detected_objects_image_array.append(splitted_copy)

                    if (output_type == "file"):
                        pltimage.imsave(output_image_path, detected_copy)

                    if (extract_detected_objects == True):
                        if (output_type == "file"):
                            return output_objects_array, detected_objects_image_array
                        elif (output_type == "array"):
                            return detected_copy, output_objects_array, detected_objects_image_array

                    else:
                        if (output_type == "file"):
                            return output_objects_array
                        elif (output_type == "array"):
                            return detected_copy, output_objects_array
                elif (self.__modelType == "yolov3" or self.__modelType == "tinyyolov3"):

                    output_objects_array = []
                    detected_objects_image_array = []

                    if (input_type == "file"):
                        image = Image.open(input_image)
                        input_image = read_image_bgr(input_image)
                    elif (input_type == "array"):
                        image = Image.fromarray(np.uint8(input_image))
                        input_image = read_image_array(input_image)
                    elif (input_type == "stream"):
                        image = Image.open(input_image)
                        input_image = read_image_stream(input_image)

                    detected_copy = input_image
                    detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                    detected_copy2 = input_image
                    detected_copy2 = cv2.cvtColor(detected_copy2, cv2.COLOR_BGR2RGB)

                    new_image_size = (self.__yolo_model_image_size[0] - (self.__yolo_model_image_size[0] % 32),
                                      self.__yolo_model_image_size[1] - (self.__yolo_model_image_size[1] % 32))
                    boxed_image = letterbox_image(image, new_image_size)
                    image_data = np.array(boxed_image, dtype="float32")

                    image_data /= 255.
                    image_data = np.expand_dims(image_data, 0)

                    model = self.__model_collection[0]

                    out_boxes, out_scores, out_classes = self.sess.run(
                        [self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes],
                        feed_dict={
                            model.input: image_data,
                            self.__yolo_input_image_shape: [image.size[1], image.size[0]],
                            K.learning_phase(): 0
                        })

                    min_probability = minimum_percentage_probability / 100
                    counting = 0

                    for a, b in reversed(list(enumerate(out_classes))):
                        predicted_class = self.numbers_to_names[b]
                        box = out_boxes[a]
                        score = out_scores[a]

                        if score < min_probability:
                            continue

                        counting += 1

                        objects_dir = output_image_path + "-objects"
                        if (extract_detected_objects == True and output_type == "file"):
                            if (os.path.exists(objects_dir) == False):
                                os.mkdir(objects_dir)

                        label = "{} {:.2f}".format(predicted_class, score)

                        top, left, bottom, right = box
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                        try:
                            color = label_color(b)
                        except:
                            color = (255, 0, 0)

                        detection_details = (left, top, right, bottom)
                        draw_box(detected_copy, detection_details, color=color)

                        if (display_object_name == True and display_percentage_probability == True):
                            draw_caption(detected_copy, detection_details, label)
                        elif (display_object_name == True):
                            draw_caption(detected_copy, detection_details, predicted_class)
                        elif (display_percentage_probability == True):
                            draw_caption(detected_copy, detection_details, str(score * 100))

                        each_object_details = {}
                        each_object_details["name"] = predicted_class
                        each_object_details["percentage_probability"] = score * 100
                        each_object_details["box_points"] = detection_details

                        output_objects_array.append(each_object_details)

                        if (extract_detected_objects == True):
                            splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
                                            detection_details[0]:detection_details[2]]
                            if (output_type == "file"):
                                splitted_image_path = os.path.join(objects_dir,
                                                                   predicted_class + "-" + str(
                                                                       counting) + ".jpg")
                                pltimage.imsave(splitted_image_path, splitted_copy)
                                detected_objects_image_array.append(splitted_image_path)
                            elif (output_type == "array"):
                                detected_objects_image_array.append(splitted_copy)

                    if (output_type == "file"):
                        pltimage.imsave(output_image_path, detected_copy)

                    if (extract_detected_objects == True):
                        if (output_type == "file"):
                            return output_objects_array, detected_objects_image_array
                        elif (output_type == "array"):
                            return detected_copy, output_objects_array, detected_objects_image_array

                    else:
                        if (output_type == "file"):
                            return output_objects_array
                        elif (output_type == "array"):
                            return detected_copy, output_objects_array
            except:
                raise ValueError(
                    "Ensure you specified correct input image, input type, output type and/or output image path ")

    def CustomObjects(self, person=False, bicycle=False, car=False, motorcycle=False, airplane=False,
                      bus=False, train=False, truck=False, boat=False, traffic_light=False, fire_hydrant=False,
                      stop_sign=False,
                      parking_meter=False, bench=False, bird=False, cat=False, dog=False, horse=False, sheep=False,
                      cow=False, elephant=False, bear=False, zebra=False,
                      giraffe=False, backpack=False, umbrella=False, handbag=False, tie=False, suitcase=False,
                      frisbee=False, skis=False, snowboard=False,
                      sports_ball=False, kite=False, baseball_bat=False, baseball_glove=False, skateboard=False,
                      surfboard=False, tennis_racket=False,
                      bottle=False, wine_glass=False, cup=False, fork=False, knife=False, spoon=False, bowl=False,
                      banana=False, apple=False, sandwich=False, orange=False,
                      broccoli=False, carrot=False, hot_dog=False, pizza=False, donot=False, cake=False, chair=False,
                      couch=False, potted_plant=False, bed=False,
                      dining_table=False, toilet=False, tv=False, laptop=False, mouse=False, remote=False,
                      keyboard=False, cell_phone=False, microwave=False,
                      oven=False, toaster=False, sink=False, refrigerator=False, book=False, clock=False, vase=False,
                      scissors=False, teddy_bear=False, hair_dryer=False,
                      toothbrush=False):

        """
                         The 'CustomObjects()' function allows you to handpick the type of objects you want to detect
                         from an image. The objects are pre-initiated in the function variables and predefined as 'False',
                         which you can easily set to true for any number of objects available.  This function
                         returns a dictionary which must be parsed into the 'detectCustomObjectsFromImage()'. Detecting
                          custom objects only happens when you call the function 'detectCustomObjectsFromImage()'


                        * true_values_of_objects (array); Acceptable values are 'True' and False  for all object values present

                        :param boolean_values:
                        :return: custom_objects_dict
                """

        custom_objects_dict = {}
        input_values = [person, bicycle, car, motorcycle, airplane,
                        bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
                        parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
                        giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
                        sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket,
                        bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
                        broccoli, carrot, hot_dog, pizza, donot, cake, chair, couch, potted_plant, bed,
                        dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave,
                        oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer,
                        toothbrush]
        actual_labels = ["person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                         "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                         "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donot", "cake", "chair", "couch", "potted plant",
                         "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair dryer",
                         "toothbrush"]

        for input_value, actual_label in zip(input_values, actual_labels):
            if (input_value == True):
                custom_objects_dict[actual_label] = "valid"
            else:
                custom_objects_dict[actual_label] = "invalid"

        return custom_objects_dict

    def detectCustomObjectsFromImage(self, custom_objects=None, input_image="", output_image_path="", input_type="file",
                                     output_type="file", extract_detected_objects=False,
                                     minimum_percentage_probability=50, display_percentage_probability=True,
                                     display_object_name=True):
        """
                    'detectCustomObjectsFromImage()' function is used to detect predefined objects observable in the given image path:
                            * custom_objects , an instance of the CustomObject class to filter which objects to detect
                            * input_image , which can be file to path, image numpy array or image file stream
                            * output_image_path , file path to the output image that will contain the detection boxes and label, if output_type="file"
                            * input_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file", "array" and "stream"
                            * output_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file" and "array"
                            * extract_detected_objects (optional, False by default) , option to save each object detected individually as an image and return an array of the objects' image path.
                            * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
                            * display_percentage_probability (optional, True by default), option to show or hide the percentage probability of each object in the saved/returned detected image
                            * display_display_object_name (optional, True by default), option to show or hide the name of each object in the saved/returned detected image

                    The values returned by this function depends on the parameters parsed. The possible values returnable
            are stated as below
            - If extract_detected_objects = False or at its default value and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (tuple of x1,y1,x2 and y2 coordinates)

            - If extract_detected_objects = False or at its default value and output_type = 'array' ,
              Then the function will return:

                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (tuple of x1,y1,x2 and y2 coordinates)

            - If extract_detected_objects = True and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (tuple of x1,y1,x2 and y2 coordinates)
                2. an array of string paths to the image of each object extracted from the image

            - If extract_detected_objects = True and output_type = 'array', the the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (tuple of x1,y1,x2 and y2 coordinates)
                3. an array of numpy arrays of each object detected in the image


            :param input_image:
            :param output_image_path:
            :param input_type:
            :param output_type:
            :param extract_detected_objects:
            :param minimum_percentage_probability:
            :return output_objects_array:
            :param display_percentage_probability:
            :param display_object_name
            :return detected_copy:
            :return detected_detected_objects_image_array:
                """

        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making object detection.")
        # elif (self.__modelLoaded == True):
        try:
            if (self.__modelType == "retinanet"):
                return self.detectCustomObjectsFromImage_retinanet(custom_objects, input_image, output_image_path, input_type,
                        output_type, extract_detected_objects, minimum_percentage_probability,
                        display_percentage_probability, display_object_name)
            elif (self.__modelType == "yolov3" or self.__modelType == "tinyyolov3"):
                # return self.detectCustomObjectsFromImage_yolo(custom_objects, input_image, output_image_path, input_type,
                #         output_type, extract_detected_objects, minimum_percentage_probability,
                #         display_percentage_probability, display_object_name)
                return self.detectCustomObjectsFromImage_yolo_simplify(custom_objects, input_image, output_image_path, input_type,
                        output_type, extract_detected_objects, minimum_percentage_probability,
                        display_percentage_probability, display_object_name)
        except:
            raise ValueError(
                "Ensure you specified correct input image, input type, output type and/or output image path ")



    def detectCustomObjectsFromImage_retinanet(self, custom_objects=None, input_image="", output_image_path="", input_type="file",
                                     output_type="file", extract_detected_objects=False,
                                     minimum_percentage_probability=50, display_percentage_probability=True,
                                     display_object_name=True):
        output_objects_array = []
        detected_objects_image_array = []

        if (input_type == "file"):
            image = read_image_bgr(input_image)
        elif (input_type == "array"):
            image = read_image_array(input_image)
        elif (input_type == "stream"):
            image = read_image_stream(input_image)

        detected_copy = image.copy()
        detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

        detected_copy2 = image.copy()
        detected_copy2 = cv2.cvtColor(detected_copy2, cv2.COLOR_BGR2RGB)

        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=self.__input_image_min, max_side=self.__input_image_max)

        model = self.__model_collection[0]
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
        predicted_numbers = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_numbers]

        detections[0, :, :4] /= scale

        min_probability = minimum_percentage_probability / 100
        counting = 0

        for index, (label, score), in enumerate(zip(predicted_numbers, scores)):
            if score < min_probability:
                continue

            if (custom_objects != None):
                check_name = self.numbers_to_names[label]
                if (custom_objects[check_name] == "invalid"):
                    continue

            counting += 1

            objects_dir = output_image_path + "-objects"
            if (extract_detected_objects == True and output_type == "file"):
                if (os.path.exists(objects_dir) == False):
                    os.mkdir(objects_dir)

            color = label_color(label)

            detection_details = detections[0, index, :4].astype(int)
            draw_box(detected_copy, detection_details, color=color)

            if (display_object_name == True and display_percentage_probability == True):
                caption = "{} {:.3f}".format(self.numbers_to_names[label], (score * 100))
                draw_caption(detected_copy, detection_details, caption)
            elif (display_object_name == True):
                caption = "{} ".format(self.numbers_to_names[label])
                draw_caption(detected_copy, detection_details, caption)
            elif (display_percentage_probability == True):
                caption = " {:.3f}".format((score * 100))
                draw_caption(detected_copy, detection_details, caption)

            each_object_details = {}
            each_object_details["name"] = self.numbers_to_names[label]
            each_object_details["percentage_probability"] = score * 100
            each_object_details["box_points"] = detection_details

            output_objects_array.append(each_object_details)

            if (extract_detected_objects == True):
                splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
                                detection_details[0]:detection_details[2]]
                if (output_type == "file"):
                    splitted_image_path = os.path.join(objects_dir,
                                                        self.numbers_to_names[label] + "-" + str(
                                                            counting) + ".jpg")
                    pltimage.imsave(splitted_image_path, splitted_copy)
                    detected_objects_image_array.append(splitted_image_path)
                elif (output_type == "array"):
                    detected_objects_image_array.append(splitted_copy)

        if (output_type == "file"):
            pltimage.imsave(output_image_path, detected_copy)

        if (extract_detected_objects == True):
            if (output_type == "file"):
                return output_objects_array, detected_objects_image_array
            elif (output_type == "array"):
                return detected_copy, output_objects_array, detected_objects_image_array

        else:
            if (output_type == "file"):
                return output_objects_array
            elif (output_type == "array"):
                return detected_copy, output_objects_array
        pass



    def detectCustomObjectsFromImage_yolo(self, custom_objects=None, input_image="", output_image_path="", input_type="file",
                                     output_type="file", extract_detected_objects=False,
                                     minimum_percentage_probability=50, display_percentage_probability=True,
                                     display_object_name=True):

        if (input_type == "file"):
            image = Image.open(input_image)
            input_image = read_image_bgr(input_image)
        elif (input_type == "array"):
            image = Image.fromarray(np.uint8(input_image))
            input_image = read_image_array(input_image)
        elif (input_type == "stream"):
            image = Image.open(input_image)
            input_image = read_image_stream(input_image)

        detected_copy = input_image
        detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

        detected_copy2 = input_image
        detected_copy2 = cv2.cvtColor(detected_copy2, cv2.COLOR_BGR2RGB)

        new_image_size = (self.__yolo_model_image_size[0] - (self.__yolo_model_image_size[0] % 32),
                            self.__yolo_model_image_size[1] - (self.__yolo_model_image_size[1] % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        model = self.__model_collection[0]

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes],
            feed_dict={
                model.input: image_data,
                self.__yolo_input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        min_probability = minimum_percentage_probability / 100
        counting = 0
        output_objects_array = []
        detected_objects_image_array = []
        for a, b in reversed(list(enumerate(out_classes))):
            predicted_class = self.numbers_to_names[b]
            box = out_boxes[a]
            score = out_scores[a]

            if score < min_probability:
                continue

            if (custom_objects != None):
                if (custom_objects[predicted_class] == "invalid"):
                    continue

            counting += 1

            objects_dir = output_image_path + "-objects"
            if (extract_detected_objects == True and output_type == "file"):
                if (os.path.exists(objects_dir) == False):
                    os.mkdir(objects_dir)

            label = "{} {:.2f}".format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            try:
                color = label_color(b)
            except:
                color = (255, 0, 0)

            detection_details = (left, top, right, bottom)
            draw_box(detected_copy, detection_details, color=color)

            if (display_object_name == True and display_percentage_probability == True):
                draw_caption(detected_copy, detection_details, label)
            elif (display_object_name == True):
                draw_caption(detected_copy, detection_details, predicted_class)
            elif (display_percentage_probability == True):
                draw_caption(detected_copy, detection_details, str(score * 100))

            each_object_details = {}
            each_object_details["name"] = predicted_class
            each_object_details["percentage_probability"] = score * 100
            each_object_details["box_points"] = detection_details

            output_objects_array.append(each_object_details)

            if (extract_detected_objects == True):
                splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
                                detection_details[0]:detection_details[2]]
                if (output_type == "file"):
                    splitted_image_path = os.path.join(objects_dir,
                                                        predicted_class + "-" + str(
                                                            counting) + ".jpg")
                    pltimage.imsave(splitted_image_path, splitted_copy)
                    detected_objects_image_array.append(splitted_image_path)
                elif (output_type == "array"):
                    detected_objects_image_array.append(splitted_copy)

        if (output_type == "file"):
            pltimage.imsave(output_image_path, detected_copy)

        if (extract_detected_objects == True):
            if (output_type == "file"):
                return output_objects_array, detected_objects_image_array
            elif (output_type == "array"):
                return detected_copy, output_objects_array, detected_objects_image_array

        else:
            if (output_type == "file"):
                return output_objects_array
            elif (output_type == "array"):
                return detected_copy, output_objects_array
        pass



    def detectCustomObjectsFromImage_yolo_simplify(self, custom_objects=None, input_image="", output_image_path="", input_type="file",
                                     output_type="file", extract_detected_objects=False,
                                     minimum_percentage_probability=50, display_percentage_probability=True,
                                     display_object_name=True):
        '''
        delete:
            extract_detected_objects
            output_type
            display_object_name
            display_percentage_probability
        '''
        try:
            if (input_type == "file"):
                image = Image.open(input_image)
                input_image = read_image_bgr(input_image)
            elif (input_type == "array"):
                image = Image.fromarray(np.uint8(input_image))
                input_image = read_image_array(input_image)
            elif (input_type == "stream"):
                image = Image.open(input_image)
                input_image = read_image_stream(input_image)
        except Exception:
            print('************************1***********************************')
            print(traceback.print_exc())
            print('***********************************************************')
            return None, None

        detected_copy = input_image
        detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

        # detected_copy2 = input_image
        # detected_copy2 = cv2.cvtColor(detected_copy2, cv2.COLOR_BGR2RGB)

        new_image_size = (self.__yolo_model_image_size[0] - (self.__yolo_model_image_size[0] % 32),
                            self.__yolo_model_image_size[1] - (self.__yolo_model_image_size[1] % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        model = self.__model_collection[0]

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes],
            feed_dict={
                model.input: image_data,
                self.__yolo_input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        min_probability = minimum_percentage_probability / 100
        counting = 0
        output_objects_array = []
        # detected_objects_image_array = []
        for a, b in reversed(list(enumerate(out_classes))):
            try:
                predicted_class = self.numbers_to_names[b]
            except Exception:
                print('************************1***********************************')
                print(traceback.print_exc())
                print('***********************************************************')
                continue
            box = out_boxes[a]
            score = out_scores[a]

            if score < min_probability:
                continue
            try:
                if (custom_objects != None):
                    if (custom_objects[predicted_class] == "invalid"):
                        continue
            except Exception:
                print('************************2***********************************')
                print(traceback.print_exc())
                print('***********************************************************')
                continue
            counting += 1

            # objects_dir = output_image_path + "-objects"
            # if (extract_detected_objects == True and output_type == "file"):
            #     if (os.path.exists(objects_dir) == False):
            #         os.mkdir(objects_dir)

            label = "{} {:.2f}".format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            try:
                color = label_color(b)
            except:
                color = (255, 0, 0)

            detection_details = (left, top, right, bottom)
            # draw_box(detected_copy, detection_details, color=color)

            # if (display_object_name == True and display_percentage_probability == True):
            #     draw_caption(detected_copy, detection_details, label)
            # elif (display_object_name == True):
            #     draw_caption(detected_copy, detection_details, predicted_class)
            # elif (display_percentage_probability == True):
            #     draw_caption(detected_copy, detection_details, str(score * 100))

            each_object_details = {}
            each_object_details["name"] = predicted_class
            each_object_details["percentage_probability"] = score * 100
            each_object_details["box_points"] = detection_details

            output_objects_array.append(each_object_details)

            # if (extract_detected_objects == True):
            #     splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
            #                     detection_details[0]:detection_details[2]]
            #     if (output_type == "file"):
            #         splitted_image_path = os.path.join(objects_dir,
            #                                             predicted_class + "-" + str(
            #                                                 counting) + ".jpg")
            #         pltimage.imsave(splitted_image_path, splitted_copy)
            #         detected_objects_image_array.append(splitted_image_path)
            #     elif (output_type == "array"):
            #         detected_objects_image_array.append(splitted_copy)

        # if (output_type == "file"):
        #     pltimage.imsave(output_image_path, detected_copy)

        # if (extract_detected_objects == True):
        #     if (output_type == "file"):
        #         return output_objects_array, detected_objects_image_array
        #     elif (output_type == "array"):
        #         return detected_copy, output_objects_array, detected_objects_image_array

        # else:
        #     if (output_type == "file"):
        #         return output_objects_array
        #     elif (output_type == "array"):
        #         return detected_copy, output_objects_array
        if len(output_objects_array) < 1:
            # return input_image, 'None', 0.0, (0,0,1,1)
            return input_image, None
        # else:
            # obj = output_objects_array[-1]
            # x1, y1, x2, y2 = obj['box_points']
            # return input_image[y1:y2, x1:x2], obj['name'], obj['percentage_probability'], obj['box_points']
        return input_image, output_objects_array
        pass
