from itertools import count

import cv2

from imageai.Detection.keras_retinanet.models.resnet import resnet50_retinanet
from imageai.Detection.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from imageai.Detection.keras_retinanet.utils.visualization import draw_box, draw_caption
from imageai.Detection.keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import numpy as np
import tensorflow as tf
import os


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class ObjectDetection:
    """
                    This is the object detection class for images in the ImageAI library. It provides support for RetinaNet
                     object detection network . After instantiating this class, you can set it's properties and
                     make object detections using it's pre-defined functions.

                     The following functions are required to be called before object detection can be made
                     * setModelPath()
                     * At least of of the following and it must correspond to the model set in the setModelPath()
                      [setModelTypeAsRetinaNet(), setModelTypeAsYOLO()]
                     * loadModel() [This must be called once only before performing object detection]

                     Once the above functions have been called, you can call the detectObjectsFromImage() function of the object detection instance
                     object at anytime to obtain observable objects in any image.
    """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.__modelPathAdded = False
        self.__modelLoaded = False
        self.__model_collection = []

        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
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
                   79: 'toothbrush'}




    def setModelTypeAsRetinaNet(self):
        """
        'setModelTypeAsRetinaNet()' is used to set the model type to the RetinaNet model
        for the object detection instance instance object .
        :return:
        """
        self.__modelType = "retinanet"


    def setModelPath(self, model_path):
        """
         'setModelPath()' function is required and is used to set the file path to a the RetinaNet
          object detectio model trained on the COCO dataset.
          :param model_path:
          :return:
        """

        if(self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True



    def loadModel(self):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function

                :param:
                :return:
        """

        if (self.__modelLoaded == False):
            if(self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif(self.__modelType == "retinanet"):
                model = resnet50_retinanet(num_classes=80)
                model.load_weights(self.modelPath)
                self.__model_collection.append(model)
                self.__modelLoaded = True


    def detectObjectsFromImage(self, image_path, output_image_path, save_detected_objects = False, minimum_percentage_probability = 50):
        """
            'detectObjectsFromImage()' function is used to detect objects observable in the given image path:
                    * image_path , file path to the image
                    * output_image_path , file path to the output image that will contain the detection boxes and label
                    * save_detected_objects (optional, False by default) , option to save each object detected individually as an image and return an array of the objects' image path.
                    * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.

            This function returns an array of dictionaries, with each dictionary corresponding to the objects
            detected in the image. Each dictionary contains the following property:
            - name
            - percentage_probability

            If 'save_detected_objects' is set to 'True', this function will return another array (making 2 arrays
            that will be returned) that contains list of all the paths to the saved image of each object detected

            :param image_path:
            :param output_image_path:
            :param save_detected_objects:
            :param minimum_percentage_probability:
            :return output_objects_array:
        """

        if(self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making object detection.")
        elif(self.__modelLoaded == True):
            try:
                output_objects_array = []
                detected_objects_image_array = []

                image = read_image_bgr(image_path)
                detected_copy = image.copy()
                detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                detected_copy2 = image.copy()
                detected_copy2 = cv2.cvtColor(detected_copy2, cv2.COLOR_BGR2RGB)

                image = preprocess_image(image)
                image, scale = resize_image(image)

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
                    if(save_detected_objects == True):
                        if (os.path.exists(objects_dir) == False):
                            os.mkdir(objects_dir)

                    color = label_color(label)

                    detection_details = detections[0, index, :4].astype(int)
                    draw_box(detected_copy, detection_details, color=color)

                    caption = "{} {:.3f}".format(self.numbers_to_names[label], (score * 100))
                    draw_caption(detected_copy, detection_details, caption)

                    each_object_details = {}
                    each_object_details["name"] = self.numbers_to_names[label]
                    each_object_details["percentage_probability"] = str(score * 100)
                    output_objects_array.append(each_object_details)

                    if(save_detected_objects == True):
                        splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
                                        detection_details[0]:detection_details[2]]
                        splitted_image_path = objects_dir + "\\" + self.numbers_to_names[label] + "-" + str(counting) + ".jpg"
                        pltimage.imsave(splitted_image_path, splitted_copy)
                        detected_objects_image_array.append(splitted_image_path)

                pltimage.imsave(output_image_path, detected_copy)

                if(save_detected_objects == True):
                    return output_objects_array, detected_objects_image_array
                else:
                    return output_objects_array
            except:
                raise ValueError("Ensure you specified correct paths for the input and output image ")


