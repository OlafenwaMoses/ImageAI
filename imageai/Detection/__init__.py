"""

"""
import os
import colorsys
import traceback


from imageai.Detection.keras_retinanet.models.resnet import resnet50_retinanet
from imageai.Detection.keras_retinanet.utils.image import read_image_bgr, read_image_array, read_image_stream, \
    preprocess_image, resize_image
from imageai.Detection.keras_retinanet.utils.visualization import draw_box, draw_caption
from imageai.Detection.keras_retinanet.utils.colors import label_color

import cv2
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import matplotlib.image as pltimage
from keras import backend as K
from keras.layers import Input
from PIL import Image

from imageai.Detection.YOLOv3.models import yolo_main, tiny_yolo_main
from imageai.Detection.YOLOv3.utils import letterbox_image, yolo_eval


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
     * set_model_path()
     * At least of of the following and it must correspond to the model set in the set_model_path()
      [set_model_type_as_retina_net(), set_model_type_as_yolo_v3(), set_model_type_as_tiny_yolo_v3()]
     * load_model() [This must be called once only before performing object detection]

     Once the above functions have been called, you can call the detect_objects() function of
     the object detection instance object at anytime to obtain observable objects in any image.
    """

    def __init__(self):
        self._model_type = ""
        self.model_path = ""
        self._model_path_added = False
        self._model_loaded = False
        self._model_collection = []

        # Instance variables for RetinaNet Model
        self._input_image_min = 1333
        self._input_image_max = 800

        self.numbers_to_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
            7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
            13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
            21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
            28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
            34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
            53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
            73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }

        # Unique instance variables for YOLOv3 and TinyYOLOv3 model
        self.__yolo_iou = 0.45
        self.__yolo_score = 0.1
        self.__yolo_anchors = np.array([[10., 13.], [16., 30.], [33., 23.], [30., 61.], [62., 45.], [59., 119.],
                                        [116., 90.], [156., 198.], [373., 326.]])
        self._yolo_model_image_size = (416, 416)
        self._yolo_boxes, self._yolo_scores, self._yolo_classes = "", "", ""
        self.sess = K.get_session()

        # Unique instance variables for TinyYOLOv3.
        self.__tiny_yolo_anchors = np.array(
            [[10., 14.], [23., 27.], [37., 58.], [81., 82.], [135., 169.], [344., 319.]])

    def set_model_type_as_retina_net(self):
        """
        'set_model_type_as_retina_net()' is used to set the model type to the RetinaNet model
        for the video object detection instance instance object.
        """
        self._model_type = "retinanet"

    def set_model_type_as_yolo_v3(self):
        """
        'set_model_type_as_yolo_v3()' is used to set the model type to the YOLOv3 model
        for the video object detection instance instance object.
        """

        self._model_type = "yolov3"

    def set_model_type_as_tiny_yolo_v3(self):
        """
        'set_model_type_as_tiny_yolo_v3()' is used to set the model type to the TinyYOLOv3 model
        for the video object detection instance instance object.
        """

        self._model_type = "tinyyolov3"

    def set_model_path(self, model_path):
        """'set_model_path()' function is required and is used to set the file path to a RetinaNet
        object detection model trained on the COCO dataset.

        :param str model_path:
        """

        if not self._model_path_added:
            self.model_path = model_path
            self._model_path_added = True

    def __load_model_yolo3(self, model, yolo_anchors):
        """ loading the Yolo type model

        :param model:
        :param yolo_anchors:
        :return: model
        """
        model.load_weights(self.model_path)

        hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                      for x in range(len(self.numbers_to_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self._yolo_input_image_shape = K.placeholder(shape=(2,))
        self._yolo_boxes, self._yolo_scores, self._yolo_classes = yolo_eval(model.output,
                                                                            yolo_anchors,
                                                                            len(self.numbers_to_names),
                                                                            self._yolo_input_image_shape,
                                                                            score_threshold=self.__yolo_score,
                                                                            iou_threshold=self.__yolo_iou)
        return model

    def load_model(self, detection_speed="normal"):
        """
        'load_model()' function is required and is used to load the model structure into the program from the file path defined
        in the set_model_path() function. This function receives an optional value which is "detection_speed".
        The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
         with just slight reduction in the number of objects detected.


        * prediction_speed (optional); Acceptable values are "normal", "fast", "faster", "fastest" and "flash"

        :param str detection_speed:
        """

        if self._model_type == "retinanet":
            input_image_nim_max = {
                'normal': (800, 1333),
                'fast': (400, 700),
                'faster': (300, 500),
                'fastest': (200, 350),
                'flash': (100, 250),
            }
            self._input_image_min, self._input_image_max = input_image_nim_max.get(detection_speed, (0, 0))

        elif self._model_type == "yolov3":
            image_size = {
                'normal':(416, 416),
                'fast':(320, 320),
                'faster':(208, 208),
                'fastest':(128, 128),
                'flash':(96, 96),
            }
            self._yolo_model_image_size = image_size.get(detection_speed, (0, 0))

        elif self._model_type == "tinyyolov3":
            image_size = {
                'normal': (832, 832),
                'fast': (576, 576),
                'faster': (416, 416),
                'fastest': (320, 320),
                'flash': (272, 272),
            }
            self._yolo_model_image_size = image_size.get(detection_speed, (0, 0))

        if not self._model_loaded:
            if self._model_type == "":
                raise ValueError("You must set a valid model type before loading the model.")
            elif self._model_type == "retinanet":
                model = resnet50_retinanet(num_classes=80)
                model.load_weights(self.model_path)
            elif self._model_type == "yolov3":
                model = yolo_main(Input(shape=(None, None, 3)), len(self.__yolo_anchors) // 3,
                                  len(self.numbers_to_names))
                model = self.__load_model_yolo3(model, self.__yolo_anchors)
            elif self._model_type == "tinyyolov3":
                model = tiny_yolo_main(Input(shape=(None, None, 3)), len(self.__tiny_yolo_anchors) // 2,
                                       len(self.numbers_to_names))
                model = self.__load_model_yolo3(model, self.__tiny_yolo_anchors)
            else:
                model = None

            # in case thet the model was successfully created
            if model:
                self._model_collection.append(model)
                self._model_loaded = True

    def __detect_objects_retinanet(self, image, output_image_path="", output_type="file",
                                   extract_detected_objects=False, minimum_percentage_probability=50,
                                   display_percentage_probability=True, display_object_name=True,
                                   custom_objects=None):
        output_objects_array = []
        detected_objects_image_array = []

        detected_copy = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

        detected_copy2 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=self._input_image_min, max_side=self._input_image_max)

        model = self._model_collection[0]
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
        predicted_numbers = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_numbers]

        detections[0, :, :4] /= scale

        min_probability = minimum_percentage_probability / 100
        counting = 0

        for index, (label, score), in enumerate(zip(predicted_numbers, scores)):
            if score < min_probability:
                continue

            if custom_objects is not None:
                check_name = self.numbers_to_names[label]
                if custom_objects[check_name] == "invalid":
                    continue

            counting += 1

            objects_dir = output_image_path + "-objects"
            if extract_detected_objects and output_type == "file" and not os.path.exists(objects_dir):
                os.mkdir(objects_dir)

            color = label_color(label)

            detection_details = detections[0, index, :4].astype(int)
            draw_box(detected_copy, detection_details, color=color)

            if display_object_name and display_percentage_probability:
                caption = "{} {:.3f}".format(self.numbers_to_names[label], (score * 100))
                draw_caption(detected_copy, detection_details, caption)
            elif display_object_name:
                caption = "{} ".format(self.numbers_to_names[label])
                draw_caption(detected_copy, detection_details, caption)
            elif display_percentage_probability:
                caption = " {:.3f}".format((score * 100))
                draw_caption(detected_copy, detection_details, caption)

            each_object_details = {
                "name": self.numbers_to_names[label],
                "percentage_probability":score * 100,
                "box_points": detection_details,
            }

            output_objects_array.append(each_object_details)

            if extract_detected_objects:
                splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
                                detection_details[0]:detection_details[2]]
                if output_type == "file":
                    splitted_image_path = os.path.join(objects_dir,
                                                       self.numbers_to_names[label] + "-" + str(
                                                           counting) + ".jpg")
                    pltimage.imsave(splitted_image_path, splitted_copy)
                    detected_objects_image_array.append(splitted_image_path)
                elif output_type == "array":
                    detected_objects_image_array.append(splitted_copy)

        if output_type == "file":
            pltimage.imsave(output_image_path, detected_copy)

        if extract_detected_objects:
            if output_type == "file":
                return output_objects_array, detected_objects_image_array
            elif output_type == "array":
                return detected_copy, output_objects_array, detected_objects_image_array

        else:
            if output_type == "file":
                return output_objects_array
            elif output_type == "array":
                return detected_copy, output_objects_array

    def __detect_objects_yolo3(self, image, output_image_path="", output_type="file",
                               extract_detected_objects=False, minimum_percentage_probability=50,
                               display_percentage_probability=True, display_object_name=True,
                               custom_objects=None):
        output_objects_array = []
        detected_objects_image_array = []

        detected_copy = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

        detected_copy2 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

        new_image_size = (self._yolo_model_image_size[0] - (self._yolo_model_image_size[0] % 32),
                          self._yolo_model_image_size[1] - (self._yolo_model_image_size[1] % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        model = self._model_collection[0]

        # if the input image is just ndarray, convert it
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self._yolo_boxes, self._yolo_scores, self._yolo_classes],
            feed_dict={
                model.input: image_data,
                self._yolo_input_image_shape: (image.size[1], image.size[0]),
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

            if custom_objects is not None:
                if custom_objects[predicted_class] == "invalid":
                    continue

            counting += 1

            objects_dir = output_image_path + "-objects"
            if extract_detected_objects and output_type == "file":
                if not os.path.exists(objects_dir):
                    os.mkdir(objects_dir)

            label = "{} {:.2f}".format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            color = label_color(b)

            detection_details = (left, top, right, bottom)
            draw_box(detected_copy, detection_details, color=color)

            if display_object_name and display_percentage_probability:
                draw_caption(detected_copy, detection_details, label)
            elif display_object_name:
                draw_caption(detected_copy, detection_details, predicted_class)
            elif display_percentage_probability:
                draw_caption(detected_copy, detection_details, str(score * 100))

            each_object_details = {
                "name": predicted_class,
                "percentage_probability":score * 100,
                "box_points": detection_details,
            }

            output_objects_array.append(each_object_details)

            if extract_detected_objects:
                splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
                                detection_details[0]:detection_details[2]]
                if output_type == "file":
                    splitted_image_path = os.path.join(
                        objects_dir, predicted_class + "-" + str(counting) + ".jpg")
                    pltimage.imsave(splitted_image_path, splitted_copy)
                    detected_objects_image_array.append(splitted_image_path)
                elif output_type == "array":
                    detected_objects_image_array.append(splitted_copy)

        if output_type == "file":
            pltimage.imsave(output_image_path, detected_copy)

        if extract_detected_objects:
            if output_type == "file":
                return output_objects_array, detected_objects_image_array
            elif output_type == "array":
                return detected_copy, output_objects_array, detected_objects_image_array

        else:
            if output_type == "file":
                return output_objects_array
            elif output_type == "array":
                return detected_copy, output_objects_array

    def detect_objects(self, input_source="", output_image_path="", input_type="file", output_type="file",
                       extract_detected_objects=False, minimum_percentage_probability=50,
                       display_percentage_probability=True, display_object_name=True):
        """
        'detect_objects()' function is used to detect objects observable in the given image path:
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


        :param input_source:
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
        return self.__detect_objects_from_image(input_source, output_image_path, input_type, output_type,
                                                extract_detected_objects, minimum_percentage_probability,
                                                display_percentage_probability, display_object_name)

    def __detect_objects_from_image(self, input_source="", output_image_path="", input_type="file", output_type="file",
                                    extract_detected_objects=False, minimum_percentage_probability=50,
                                    display_percentage_probability=True, display_object_name=True, custom_objects=None):
        """
        '__detect_objects_from_image()' function is used to detect objects observable in the given image path:
            * input_image , which can be file to path, image numpy array or image file stream
            * output_image_path (only if output_type = file) , file path to the output image that will contain the detection boxes and label, if output_type="file"
            * input_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file", "array" and "stream"
            * output_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file" and "array"
            * extract_detected_objects (optional) , option to save each object detected individually as an image and return an array of the objects' image path.
            * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
            * display_percentage_probability (optional, True by default), option to show or hide the percentage probability of each object in the saved/returned detected image
            * display_display_object_name (optional, True by default), option to show or hide the name of each object in the saved/returned detected image
            * custom_objects , an instance of the CustomObject class to filter which objects to detect


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


        :param input_source:
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
        if input_type == "file":
            image = read_image_bgr(input_source)
        elif input_type == "array":
            image = read_image_array(input_source)
        elif input_type == "stream":
            image = read_image_stream(input_source)
        else:
            raise ValueError('Unsupported input type: %s', input_type)

        if not self._model_loaded:
            raise ValueError("You must call the load_model() function before making object detection.")
        elif self._model_loaded:
            try:
                if self._model_type == "retinanet":
                    result = self.__detect_objects_retinanet(image, output_image_path, output_type,
                                                             extract_detected_objects, minimum_percentage_probability,
                                                             display_percentage_probability, display_object_name,
                                                             custom_objects)
                    return result
                elif self._model_type in ("yolov3", "tinyyolov3"):
                    result = self.__detect_objects_yolo3(image, output_image_path, output_type,
                                                         extract_detected_objects, minimum_percentage_probability,
                                                         display_percentage_probability, display_object_name,
                                                         custom_objects)
                    return result
                else:
                    raise ValueError('Missing selected model type: %r' % getattr(self, '_model_type', None))

            except:
                traceback.print_exc()
                # raise ValueError(
                #     "Ensure you specified correct input image, input type, output type and/or output image path")

    @staticmethod
    def custom_objects(person=False, bicycle=False, car=False, motorcycle=False, airplane=False,
                       bus=False, train=False, truck=False, boat=False, traffic_light=False, fire_hydrant=False,
                       stop_sign=False, parking_meter=False, bench=False, bird=False, cat=False, dog=False, horse=False,
                       sheep=False, cow=False, elephant=False, bear=False, zebra=False, giraffe=False, backpack=False,
                       umbrella=False, handbag=False, tie=False, suitcase=False, frisbee=False, skis=False,
                       snowboard=False, sports_ball=False, kite=False, baseball_bat=False, baseball_glove=False,
                       skateboard=False, surfboard=False, tennis_racket=False, bottle=False, wine_glass=False, cup=False,
                       fork=False, knife=False, spoon=False, bowl=False, banana=False, apple=False, sandwich=False,
                       orange=False, broccoli=False, carrot=False, hot_dog=False, pizza=False, donot=False, cake=False,
                       chair=False, couch=False, potted_plant=False, bed=False, dining_table=False, toilet=False,
                       tv=False, laptop=False, mouse=False, remote=False, keyboard=False, cell_phone=False,
                       microwave=False, oven=False, toaster=False, sink=False, refrigerator=False, book=False,
                       clock=False, vase=False, scissors=False, teddy_bear=False, hair_dryer=False, toothbrush=False):

        """
         The 'custom_objects()' function allows you to handpick the type of objects you want to detect
         from an image. The objects are pre-initiated in the function variables and predefined as 'False',
         which you can easily set to true for any number of objects available.  This function
         returns a dictionary which must be parsed into the 'detect_custom_objects()'. Detecting
          custom objects only happens when you call the function 'detect_custom_objects()'


        * true_values_of_objects (array); Acceptable values are 'True' and False  for all object values present

        :param boolean_values:
        :return: custom_objects_dict
        """

        custom_objects_dict = {}
        input_values = (
            person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
            parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
            umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports_ball, kite, baseball_bat, baseball_glove,
            skateboard, surfboard, tennis_racket, bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple,
            sandwich, orange, broccoli, carrot, hot_dog, pizza, donot, cake, chair, couch, potted_plant, bed,
            dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave, oven, toaster, sink,
            refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer, toothbrush
        )
        actual_labels = (
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donot", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair dryer", "toothbrush"
        )

        for input_value, actual_label in zip(input_values, actual_labels):
            if input_value:
                custom_objects_dict[actual_label] = "valid"
            else:
                custom_objects_dict[actual_label] = "invalid"

        return custom_objects_dict

    def detect_custom_objects(self, custom_objects=None, input_source="", output_image_path="", input_type="file",
                              output_type="file", extract_detected_objects=False,
                              minimum_percentage_probability=50, display_percentage_probability=True,
                              display_object_name=True):
        """
        'detect_custom_objects()' function is used to detect predefined objects observable in the given image path:
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


        :param input_source:
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
        return self.__detect_objects_from_image(input_source, output_image_path, input_type, output_type,
                                                extract_detected_objects, minimum_percentage_probability,
                                                display_percentage_probability, display_object_name, custom_objects)


class VideoObjectDetection(ObjectDetection):
    """
    This is the object detection class for videos and camera live stream inputs in the ImageAI library. It provides support for RetinaNet,
     YOLOv3 and TinyYOLOv3 object detection networks. After instantiating this class, you can set it's properties and
     make object detections using it's pre-defined functions.

     The following functions are required to be called before object detection can be made
     * set_model_path()
     * At least of of the following and it must correspond to the model set in the set_model_path()
      [set_model_type_as_retina_net(), set_model_type_as_yolo_v3(), setModelTinyYOLOv3()]
     * load_model() [This must be called once only before performing object detection]

     Once the above functions have been called, you can call the detect_objects() function
     or the detect_custom_objects() of  the object detection instance object at anytime to
     obtain observable objects in any video or camera live stream.
    """

    def __init__(self):
        super(VideoObjectDetection, self).__init__()
        self.__detection_storage = None

    def __detect_objects_retinanet(self, input_video, output_video, frames_per_second=20,
                                   frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False,
                                   display_percentage_probability=True, display_object_name=True, save_detected_video=True,
                                   per_frame_function=None, per_second_function=None, per_minute_function=None,
                                   video_complete_function=None, return_detected_frame=False, custom_objects=None):
        output_frames_dict = {}
        output_frames_count_dict = {}

        counting = 0
        predicted_numbers = None
        scores = None
        detections = None

        model = self._model_collection[0]

        while input_video.isOpened():
            ret, frame = input_video.read()

            if not ret:
                break

            output_objects_array = []

            counting += 1

            if log_progress:
                print("Processing Frame : ", str(counting))

            detected_copy = frame.copy()
            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

            frame = preprocess_image(frame)
            frame, scale = resize_image(frame, min_side=self._input_image_min,
                                        max_side=self._input_image_max)

            check_frame_interval = counting % frame_detection_interval

            if counting == 1 or check_frame_interval == 0:
                _, _, detections = model.predict_on_batch(np.expand_dims(frame, axis=0))
                predicted_numbers = np.argmax(detections[0, :, 4:], axis=1)
                scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_numbers]

                detections[0, :, :4] /= scale

            min_probability = minimum_percentage_probability / 100

            for index, (label, score), in enumerate(zip(predicted_numbers, scores)):
                if score < min_probability:
                    continue

                if custom_objects is not None:
                    check_name = self.numbers_to_names[label]
                    if custom_objects[check_name] == "invalid":
                        continue

                color = label_color(label)

                detection_details = detections[0, index, :4].astype(int)
                draw_box(detected_copy, detection_details, color=color)

                if display_object_name and display_percentage_probability:
                    caption = "{} {:.3f}".format(self.numbers_to_names[label], (score * 100))
                    draw_caption(detected_copy, detection_details, caption)
                elif display_object_name:
                    caption = "{} ".format(self.numbers_to_names[label])
                    draw_caption(detected_copy, detection_details, caption)
                elif display_percentage_probability:
                    caption = " {:.3f}".format((score * 100))
                    draw_caption(detected_copy, detection_details, caption)

                output_objects_array.append({
                    "name": self.numbers_to_names[label],
                    "percentage_probability": score * 100,
                    "box_points": detection_details,
                })

            output_frames_dict[counting] = output_objects_array

            output_objects_count = {}
            for each_item in output_objects_array:
                each_item_name = each_item["name"]
                try:
                    output_objects_count[each_item_name] = output_objects_count[each_item_name] + 1
                except:
                    output_objects_count[each_item_name] = 1

            output_frames_count_dict[counting] = output_objects_count

            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

            if save_detected_video:
                output_video.write(detected_copy)

            if per_frame_function is not None:
                if return_detected_frame:
                    per_frame_function(counting, output_objects_array, output_objects_count,
                                       detected_copy)
                else:
                    per_frame_function(counting, output_objects_array, output_objects_count)

            output_frames_dict, output_frames_count_dict = VideoObjectDetection.__video_per_time_function(
                per_second_function, counting, frames_per_second, detected_copy,
                output_frames_dict, output_frames_count_dict, return_detected_frame)

            output_frames_dict, output_frames_count_dict = VideoObjectDetection.__video_per_time_function(
                per_minute_function, counting, frames_per_second, detected_copy,
                output_frames_dict, output_frames_count_dict, return_detected_frame)

        VideoObjectDetection.__video_complete_function(video_complete_function, counting,
                                                       output_frames_dict, output_frames_count_dict)

    def __detect_objects_yolo3(self, input_video, output_video, frames_per_second=20,
                               frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False,
                               display_percentage_probability=True, display_object_name=True, save_detected_video=True,
                               per_frame_function=None, per_second_function=None, per_minute_function=None,
                               video_complete_function=None, return_detected_frame=False, custom_objects=None):
        output_frames_dict = {}
        output_frames_count_dict = {}

        counting = 0
        out_boxes = None
        out_scores = None
        out_classes = None

        model = self._model_collection[0]

        while input_video.isOpened():
            ret, frame = input_video.read()

            if not ret:
                break

            output_objects_array = []

            counting += 1

            if log_progress:
                print("Processing Frame : ", str(counting))

            detected_copy = frame.copy()
            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(np.uint8(frame))

            new_image_size = (self._yolo_model_image_size[0] - (self._yolo_model_image_size[0] % 32),
                              self._yolo_model_image_size[1] - (self._yolo_model_image_size[1] % 32))
            boxed_image = letterbox_image(frame, new_image_size)
            image_data = np.array(boxed_image, dtype="float32")

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)

            check_frame_interval = counting % frame_detection_interval

            if counting == 1 or check_frame_interval == 0:
                out_boxes, out_scores, out_classes = self.sess.run(
                    [self._yolo_boxes, self._yolo_scores, self._yolo_classes],
                    feed_dict={
                        model.input: image_data,
                        self._yolo_input_image_shape: [frame.size[1], frame.size[0]],
                        K.learning_phase(): 0,
                    })

            min_probability = minimum_percentage_probability / 100

            for idx, label in reversed(list(enumerate(out_classes))):
                predicted_class = self.numbers_to_names[label]
                box = out_boxes[idx]
                score = out_scores[idx]

                if score < min_probability:
                    continue

                if custom_objects is not None:
                    if custom_objects[predicted_class] == "invalid":
                        continue

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(frame.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(frame.size[0], np.floor(right + 0.5).astype('int32'))

                color = label_color(label)

                detection_details = (left, top, right, bottom)
                draw_box(detected_copy, detection_details, color=color)

                if display_object_name and display_percentage_probability:
                    draw_caption(detected_copy, detection_details,
                                 "{} {:.2f}".format(predicted_class, score))
                elif display_object_name:
                    draw_caption(detected_copy, detection_details, predicted_class)
                elif display_percentage_probability:
                    draw_caption(detected_copy, detection_details, str(score * 100))

                output_objects_array.append({
                    "name": predicted_class,
                    "percentage_probability": score * 100,
                    "box_points": detection_details,
                })

            output_frames_dict[counting] = output_objects_array

            output_objects_count = {}
            for each_item in output_objects_array:
                each_item_name = each_item["name"]
                try:
                    output_objects_count[each_item_name] = output_objects_count[each_item_name] + 1
                except:
                    output_objects_count[each_item_name] = 1

            output_frames_count_dict[counting] = output_objects_count
            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

            if save_detected_video:
                output_video.write(detected_copy)

            if per_frame_function is not None:
                if return_detected_frame:
                    per_frame_function(counting, output_objects_array, output_objects_count,
                                       detected_copy)
                else:
                    per_frame_function(counting, output_objects_array, output_objects_count)

            output_frames_dict, output_frames_count_dict = VideoObjectDetection.__video_per_time_function(
                per_second_function, counting, frames_per_second, detected_copy,
                output_frames_dict, output_frames_count_dict, return_detected_frame)

            output_frames_dict, output_frames_count_dict = VideoObjectDetection.__video_per_time_function(
                per_minute_function, counting, frames_per_second, detected_copy,
                output_frames_dict, output_frames_count_dict, return_detected_frame)

        VideoObjectDetection.__video_complete_function(video_complete_function, counting,
                                                       output_frames_dict, output_frames_count_dict)

    @staticmethod
    def __video_per_time_function(per_time_function, counting, frames_per_time, detected_copy,
                                  output_frames_dict, output_frames_count_dict, return_detected_frame):

        if per_time_function is None or counting == 0 or (counting % frames_per_time) != 0:
            return output_frames_dict, output_frames_count_dict

        this_time_output_object_array = []
        this_time_counting_array = []
        this_time_counting = {}

        for aa in range(counting):
            if aa >= (counting - frames_per_time):
                this_time_output_object_array.append(output_frames_dict[aa + 1])
                this_time_counting_array.append(output_frames_count_dict[aa + 1])

        for each_counting_dict in this_time_counting_array:
            for each_item in each_counting_dict:
                try:
                    this_time_counting[each_item] = this_time_counting[each_item] + \
                                                     each_counting_dict[each_item]
                except:
                    this_time_counting[each_item] = each_counting_dict[each_item]

        for each_counting_item in this_time_counting:
            this_time_counting[each_counting_item] = \
                this_time_counting[each_counting_item] / frames_per_time

        if return_detected_frame:
            per_time_function(int(counting / frames_per_time),
                              this_time_output_object_array, this_time_counting_array,
                              this_time_counting, detected_copy)
        else:
            per_time_function(int(counting / frames_per_time),
                              this_time_output_object_array, this_time_counting_array,
                              this_time_counting)
        return output_frames_dict, output_frames_count_dict

    @staticmethod
    def __video_complete_function(video_complete_function, counting, output_frames_dict,
                                  output_frames_count_dict):
        if video_complete_function is None:
            return output_frames_dict, output_frames_count_dict

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
            this_video_counting[eachCountingItem] = \
                this_video_counting[eachCountingItem] / counting

        video_complete_function(this_video_output_object_array, this_video_counting_array,
                                this_video_counting)
        return output_frames_dict, output_frames_count_dict

    def __detect_objects_from_video(self, input_file_path="", camera_input=None, output_file_path="", frames_per_second=20,
                                    frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False,
                                    display_percentage_probability=True, display_object_name=True, save_detected_video=True,
                                    per_frame_function=None, per_second_function=None, per_minute_function=None,
                                    video_complete_function=None, return_detected_frame=False, custom_objects=None):
        """
        :param custom_objects: is the dictionary returned by the 'custom_objects' function
        :param input_file_path: is the file path to the input video. It is required only if 'camera_input' is not set
        :param camera_input: allows you to parse in camera input for live video detections
        :param output_file_path: is the path to the output video. It is required only if 'save_detected_video' is not set to False
        :param save_detected_video:
        :param frames_per_second: is the number of frames to be used in the output video
        :param frame_detection_interval: is the intervals of frames that will be detected
        :param minimum_percentage_probability: option to set the minimum percentage probability for nominating a detected object for output
        :param log_progress: states if the progress of the frame processed is to be logged to console
        :param display_percentage_probability:
        :param display_object_name:
        :param per_frame_function:
        :param per_second_function:
        :param per_minute_function:
        :param video_complete_function:
        :param return_detected_frame:
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
        if input_file_path == "" and camera_input is None:
            raise ValueError(
                "You must set 'input_file_path' to a valid video file, or set 'camera_input' to a valid camera")

        elif save_detected_video and output_file_path == "":
            raise ValueError(
                "You must set 'output_video_filepath' to a valid video file name, in which the detected video will be saved."
                " If you don't intend to save the detected video, set 'save_detected_video=False'")

        else:
            try:
                input_video = cv2.VideoCapture(input_file_path)

                if camera_input is not None:
                    input_video = camera_input

                output_video_filepath = output_file_path + '.avi'

                frame_width = int(input_video.get(3))
                frame_height = int(input_video.get(4))
                output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                               frames_per_second,
                                               (frame_width, frame_height))

                if self._model_type == "retinanet":

                    self.__detect_objects_retinanet(input_video, output_video, frames_per_second,
                                                    frame_detection_interval, minimum_percentage_probability, log_progress,
                                                    display_percentage_probability, display_object_name, save_detected_video,
                                                    per_frame_function, per_second_function, per_minute_function,
                                                    video_complete_function, return_detected_frame, custom_objects)

                elif self._model_type in ("yolov3", "tinyyolov3"):

                    self.__detect_objects_yolo3(input_video, output_video, frames_per_second,
                                                frame_detection_interval, minimum_percentage_probability, log_progress,
                                                display_percentage_probability, display_object_name, save_detected_video,
                                                per_frame_function, per_second_function, per_minute_function,
                                                video_complete_function, return_detected_frame, custom_objects)

                else:
                    raise ValueError('Missing selected model type: %r' % getattr(self, '_model_type', None))

                input_video.release()
                output_video.release()

                if save_detected_video:
                    return output_video_filepath

            except Exception:
                traceback.print_exc()
                raise ValueError(
                    "An error occured. It may be that your input video is invalid."
                    " Ensure you specified a proper string value for 'output_file_path' is 'save_detected_video' is not False."
                    " Also ensure your per_frame, per_second, per_minute or video_complete_analysis function"
                    " is properly configured to receive the right parameters.")

    def detect_objects(self, input_file_path="", camera_input=None, output_file_path="", frames_per_second=20,
                       frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False,
                       display_percentage_probability=True, display_object_name=True, save_detected_video=True,
                       per_frame_function=None, per_second_function=None, per_minute_function=None,
                       video_complete_function=None, return_detected_frame=False):

        """
        'detect_objects()' function is used to detect objects observable in the given video path or a camera input:
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
        * per_frame_function (optional), this parameter allows you to parse in a function you will want to execute after
            each frame of the video is detected. If this parameter is set to a function, after every video
            frame is detected, the function will be executed with the following values parsed into it:
            -- position number of the frame
            -- an array of dictinaries, with each dictinary corresponding to each object detected.
                Each dictionary contains 'name', 'percentage_probability' and 'box_points'
            -- a dictionary with with keys being the name of each unique objects and value
                are the number of instances of the object present
            -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                as the fourth value into the function

        * per_second_function (optional), this parameter allows you to parse in a function you will want to execute after
            each second of the video is detected. If this parameter is set to a function, after every second of a video
             is detected, the function will be executed with the following values parsed into it:
            -- position number of the second
            -- an array of dictionaries whose keys are position number of each frame present in the last second , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

            -- an array of dictionaries, with each dictionary corresponding to each frame in the past second, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

            -- a dictionary with its keys being the name of each unique object detected throughout the past second, and the key values are the average number of instances of the object found in all the frames contained in the past second

            -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                as the fifth value into the function

        * per_minute_function (optional), this parameter allows you to parse in a function you will want to execute after
            each minute of the video is detected. If this parameter is set to a function, after every minute of a video
             is detected, the function will be executed with the following values parsed into it:
            -- position number of the minute
            -- an array of dictionaries whose keys are position number of each frame present in the last minute , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

            -- an array of dictionaries, with each dictionary corresponding to each frame in the past minute, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

            -- a dictionary with its keys being the name of each unique object detected throughout the past minute, and the key values are the average number of instances of the object found in all the frames contained in the past minute

            -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                as the fifth value into the function

        * video_complete_function (optional), this parameter allows you to parse in a function you will want to execute after
            all of the video frames have been detected. If this parameter is set to a function, after all of frames of a video
             is detected, the function will be executed with the following values parsed into it:
            -- an array of dictionaries whose keys are position number of each frame present in the entire video , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

            -- an array of dictionaries, with each dictionary corresponding to each frame in the entire video, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

            -- a dictionary with its keys being the name of each unique object detected throughout the entire video, and the key values are the average number of instances of the object found in all the frames contained in the entire video

        * return_detected_frame (optionally, False by default), option to obtain the return the last detected video frame into the per_per_frame_function,
                                                                per_per_second_function or per_per_minute_function


        :param input_file_path:
        :param camera_input
        :param output_file_path:
        :param save_detected_video:
        :param frames_per_second:
        :param frame_detection_interval:
        :param minimum_percentage_probability:
        :param log_progress:
        :param display_percentage_probability:
        :param display_object_name:
        :param per_frame_function:
        :param per_second_function:
        :param per_minute_function:
        :param video_complete_function:
        :param return_detected_frame:
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
        return self.__detect_objects_from_video(input_file_path, camera_input, output_file_path, frames_per_second,
                                                frame_detection_interval, minimum_percentage_probability, log_progress,
                                                display_percentage_probability, display_object_name, save_detected_video,
                                                per_frame_function, per_second_function, per_minute_function,
                                                video_complete_function, return_detected_frame)

    def detect_custom_objects(self, custom_objects=None, input_file_path="", camera_input=None,
                              output_file_path="", frames_per_second=20, frame_detection_interval=1,
                              minimum_percentage_probability=50, log_progress=False,
                              display_percentage_probability=True, display_object_name=True,
                              save_detected_video=True, per_frame_function=None, per_second_function=None,
                              per_minute_function=None, video_complete_function=None,
                              return_detected_frame=False):
        """
        'detect_objects()' function is used to detect specific object(s) observable in the given video path or given camera live stream input:
            * custom_objects , which is the dictionary returned by the 'custom_objects' function
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
            * per_frame_function (optional), this parameter allows you to parse in a function you will want to execute after
                each frame of the video is detected. If this parameter is set to a function, after every video
                frame is detected, the function will be executed with the following values parsed into it:
                -- position number of the frame
                -- an array of dictinaries, with each dictinary corresponding to each object detected.
                    Each dictionary contains 'name', 'percentage_probability' and 'box_points'
                -- a dictionay with with keys being the name of each unique objects and value
                    are the number of instances of the object present
                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                    as the fourth value into the function

            * per_second_function (optional), this parameter allows you to parse in a function you will want to execute after
                each second of the video is detected. If this parameter is set to a function, after every second of a video
                 is detected, the function will be executed with the following values parsed into it:
                -- position number of the second
                -- an array of dictionaries whose keys are position number of each frame present in the last second , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

                -- an array of dictionaries, with each dictionary corresponding to each frame in the past second, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

                -- a dictionary with its keys being the name of each unique object detected throughout the past second, and the key values are the average number of instances of the object found in all the frames contained in the past second

                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                    as the fifth value into the function

            * per_minute_function (optional), this parameter allows you to parse in a function you will want to execute after
                each minute of the video is detected. If this parameter is set to a function, after every minute of a video
                 is detected, the function will be executed with the following values parsed into it:
                -- position number of the minute
                -- an array of dictionaries whose keys are position number of each frame present in the last minute , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

                -- an array of dictionaries, with each dictionary corresponding to each frame in the past minute, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

                -- a dictionary with its keys being the name of each unique object detected throughout the past minute, and the key values are the average number of instances of the object found in all the frames contained in the past minute

                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                    as the fifth value into the function

            * video_complete_function (optional), this parameter allows you to parse in a function you will want to execute after
                all of the video frames have been detected. If this parameter is set to a function, after all of frames of a video
                 is detected, the function will be executed with the following values parsed into it:
                -- an array of dictionaries whose keys are position number of each frame present in the entire video , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

                -- an array of dictionaries, with each dictionary corresponding to each frame in the entire video, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

                -- a dictionary with its keys being the name of each unique object detected throughout the entire video, and the key values are the average number of instances of the object found in all the frames contained in the entire video

            * return_detected_frame (optionally, False by default), option to obtain the return the last detected video frame into the per_per_frame_function,
                                                                    per_per_second_function or per_per_minute_function


        :param custom_objects:
        :param input_file_path:
        :param camera_input
        :param output_file_path:
        :param save_detected_video:
        :param frames_per_second:
        :param frame_detection_interval:
        :param minimum_percentage_probability:
        :param log_progress:
        :param display_percentage_probability:
        :param display_object_name:
        :param per_frame_function:
        :param per_second_function:
        :param per_minute_function:
        :param video_complete_function:
        :param return_detected_frame:
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
        return self.__detect_objects_from_video(input_file_path, camera_input, output_file_path, frames_per_second,
                                                frame_detection_interval, minimum_percentage_probability, log_progress,
                                                display_percentage_probability, display_object_name, save_detected_video,
                                                per_frame_function, per_second_function, per_minute_function,
                                                video_complete_function, return_detected_frame, custom_objects)
