from imageai.Detection import ObjectDetection
import os
import cv2
from datetime import datetime
execution_path = os.getcwd()


# datetime object containing current date and time
now = datetime.now()

print("now =", now)

# dd/mm/YY H:M:S

dt = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string = dt.replace('/', '')
dt_string = dt_string.replace(' ', '')
dt_string = dt_string.replace(':', '')
print("date and time =", dt_string)
camera = cv2.VideoCapture(0)
# Using namedWindow()
# A window with 'Live-feed' name is created
# with WINDOW_AUTOSIZE, window size is set automatically
cv2.namedWindow("Live-feed", cv2.WINDOW_AUTOSIZE)


def forFrame(frame_number, output_array, output_count, detected_frame):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("Returned Objects is : ", type(detected_frame))
    print("------------END OF A FRAME --------------")


while (True):
    # Capture the video frame
    # by frame
    ret, frame = camera.read()
    # Display the resulting frame

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(
        execution_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()
    detected = os.path.join(
        execution_path, "detectedimage" + dt_string + ".png")
    custom_objects = detector.CustomObjects(
        person=True, bicycle=True, motorcycle=True)

    detections = detector.detectObjectsFromImage(input_image=frame, output_image_path=os.path.join(
        execution_path, "detectedimage" + dt_string + ".png"), minimum_percentage_probability=30)
    size = os.path.getsize(os.path.join(execution_path, detected))
    print("File size is", size)
    print("File is type", type(detected))
    if (size > 200774):
        # Reading an image in grayscale mode
        image = cv2.imread(detected)
        cv2.imshow("Live-feed", image)

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"],
              " : ", eachObject["box_points"])
        print("--------------------------------")

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
