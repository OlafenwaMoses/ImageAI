import os
from time import time

import pandas as pd

from imageai.Detection import ObjectDetection

PATH_HERE = os.getcwd()
PATH_MODEL = os.path.join(PATH_HERE, "yolo.h5")
PATH_IMAGE_INPUT = os.path.join(PATH_HERE, "image3.jpg")
PATH_IMAGE_DETECT = os.path.join(PATH_HERE, "image3new.jpg")


def main(path_model, path_img, path_detect):
    detector = ObjectDetection()
    detector.set_model_type_as_yolo_v3()
    detector.set_model_path(path_model)
    detector.load_model()

    our_time = time()
    detections = detector.detect_objects(input_source=path_img, output_image_path=path_detect,
                                         minimum_percentage_probability=30)
    print("IT TOOK : ", time() - our_time)
    print(pd.DataFrame(detections))


if __name__ == '__main__':
    main(path_model=PATH_MODEL, path_img=PATH_IMAGE_INPUT, path_detect=PATH_IMAGE_DETECT)
