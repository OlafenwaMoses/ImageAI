# ImageAI : Custom Video Object Detection, Tracking  and Analysis<br>
<p>An <b>DeepQuest AI</b> project <a href="https://deepquestai.com" >https://deepquestai.com </a></p>
<hr>
<br>
<h3><b><u>TABLE OF CONTENTS</u></b></h3>
<a href="#videodetection" >&#9635 First Custom Video Object Detection</a><br>
<a href="#camerainputs" >&#9635 Camera / Live Stream Video Detection</a><br>
<a href="#videoanalysis" >&#9635 Video Analysis</a><br>

<a href="#hidingdetails" >&#9635 Hiding/Showing Object Name and Probability</a><br>
<a href="#videodetectionintervals" >&#9635 Frame Detection Intervals</a><br>
<a href="#detectiontimeout" >&#9635 Video Detection Timeout (NEW)</a><br>
<a href="#documentation" >&#9635 Documentation</a><br>
<br>
      ImageAI provides convenient, flexible and powerful methods to perform object detection on videos using your own <b>custom YOLOv3 model</b> and the corresponding <b>detection_config.json</b> generated during the training. This version of <b>ImageAI</b> provides commercial grade video objects detection features, which include but not limited to device/IP camera inputs, per frame, per second, per minute and entire video analysis for storing in databases and/or real-time visualizations and for future insights.
To test the custom video object detection,you can download a sample custom model we have trained to detect the Hololens headset and its <b>detection_config.json</b> file via the links below:<br> <br>
 <span><b>- <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/hololens-ex-60--loss-2.76.h5" style="text-decoration: none;" >hololens-ex-60--loss-2.76.h5</a></b> <b>(Size = 236 mb) </b></span> <br>

<span><b>- <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/detection_config.json" style="text-decoration: none;" >detection_config.json</a></b> <br>


Because video object detection is a compute intensive tasks, we advise you perform this experiment using a computer with a NVIDIA GPU and the GPU version of Tensorflow
 installed. Performing Video Object Detection CPU will be slower than using an NVIDIA GPU powered computer. You can use Google Colab for this
 experiment as it has an NVIDIA K80 GPU available for free.
<br> <br>
 Once you download the custom object detection model  and JSON files, you should copy the model and the JSON files to the your project folder where your .py files will be.
 Then create a python file and give it a name; an example is FirstCustomVideoObjectDetection.py. Then write the code below into the python file: <br><br>


<div id="videodetection" ></div>
 <h3><b>FirstCustomVideoObjectDetection.py</b></h3>

<pre>from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="holo1.mp4",
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,
                                          minimum_percentage_probability=40,
                                          log_progress=True)
</pre>

<div style="width: 600px;" >
          <b><p><i>Input Video</i></p></b>
          <a href="../../../data-videos/holo1.mp4" >
<img src="../../../data-images/holo-video.jpg" />
</a>
          <b><p><i>Output Video</i></p></b>
          <a href="https://www.youtube.com/watch?v=4o5GyAR4Mpw" >
<img src="../../../data-images/holo-video-detected.jpg" />
</a><p><a href="https://www.youtube.com/watch?v=4o5GyAR4Mpw" >https://www.youtube.com/watch?v=4o5GyAR4Mpw</a></p>
    </div> <br>

<br>
Let us make a breakdown of the object detection code that we used above.
<pre>
from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()
</pre>
 In the 3 lines above , we import the <b>ImageAI custom video object detection </b> class in the first line, import the <b>os</b> in the second line and obtained
  the path to folder where our python file runs.
  <pre>
video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()
  </pre>
  In the 4 lines above, we created a new instance of the <b>CustomVideoObjectDetection</b> class in the first line, set the model type to YOLOv3 in the second line,
  set the model path to our custom YOLOv3 model file in the third line, specified the path to the model's corresponding <b>detection_config.json</b> in the fourth line and load the model in the fifth line.

  <pre>
video_detector.detectObjectsFromVideo(input_file_path="holo1.mp4",
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,
                                          minimum_percentage_probability=40,
                                          log_progress=True)
</pre>

In the code above, we ran the <b>detectObjectsFromVideo()</b> function and parse in the path to our video,the path to the new
 video (without the extension, it saves a .avi video by default) which the function will save, the number of frames per second (fps) that
 you we desire the output video to have and option to log the progress of the detection in the console. Then the function returns a the path to the saved video
 which contains boxes and percentage probabilities rendered on objects detected in the video.




<div id="camerainputs"></div>
<h3><b><u>Camera / Live Stream Video Detection</u></b></h3>
<b>ImageAI</b> now allows live-video detection with support for camera inputs. Using <b>OpenCV</b>'s <b>VideoCapture()</b> function, you can load live-video streams from a device camera, cameras connected by cable or IP cameras, and parse it into <b>ImageAI</b>'s <b>detectObjectsFromVideo()</b> function. All features that are supported for detecting objects in a video file is also available for detecting objects in a camera's live-video feed. Find below an example of detecting live-video feed from the device camera. <br>

<pre>
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()


camera = cv2.VideoCapture(0)

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,
                                          minimum_percentage_probability=40,
                                          log_progress=True)
</pre>

The difference in the code above and the code for the detection of a video file is that we defined an <b>OpenCV VideoCapture</b> instance and loaded the default device camera into it. Then we parsed the camera we defined into the parameter <b>camera_input</b> which replaces the <b>input_file_path</b> that is used for video file.

<br><br>
<div id="videoanalysis"></div>
<h3><b><u>Video Analysis</u></b></h3>

<b>ImageAI</b> now provide commercial-grade video analysis in the Custom Video Object Detection class, for both video file inputs and camera inputs. This feature allows developers to obtain deep insights into any video processed with <b>ImageAI</b>. This insights can be visualized in real-time, stored in a NoSQL database for future review or analysis. <br><br>

For video analysis, the <b>detectObjectsFromVideo()</b> now allows you to state your own defined functions which will be executed for every frame, seconds and/or minute of the video detected as well as a state a function that will be executed at the end of a video detection. Once this functions are stated, they will receive raw but comprehensive analytical data on the index of the frame/second/minute, objects detected (name, percentage_probability and box_points), number of instances of each unique object detected and average number of occurrence of each unique object detected over a second/minute and entire video. <br>
To obtain the video analysis, all you need to do is specify a function, state the corresponding parameters it will be receiving and parse the function name into the <b>per_frame_function</b>, <b>per_second_function</b>, <b>per_minute_function</b> and <b>video_complete_function</b> parameters in the detection function. Find below examples of video analysis functions. <br>

<pre>
def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")


video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20, per_second_function=forSeconds, per_frame_function = forFrame, per_minute_function= forMinute,
                                          minimum_percentage_probability=40,
                                          log_progress=True)

</pre>


<b>ImageAI</b> also allows you to obtain complete analysis of the entire video processed. All you need is to define a function like the forSecond or forMinute function and set the <b>video_complete_function</b> parameter into your <b>.detectObjectsFromVideo()</b> function. The same values for the per_second-function and per_minute_function will be returned. The difference is that no index will be returned and the other 3 values will be returned, and the 3 values will cover all frames in the video. Below is a sample function: <br>
<pre>
def forFull(output_arrays, count_arrays, average_output_count):
    #Perform action on the 3 parameters returned into the function


video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          video_complete_function=forFull,
                                          minimum_percentage_probability=40,
                                          log_progress=True)

</pre>
<br>
<b>FINAL NOTE ON VIDEO ANALYSIS</b> : <b>ImageAI</b> allows you to obtain the detected video frame as a Numpy array at each frame, second and minute function. All you need to do is specify one more parameter in your function and set <b>return_detected_frame=True</b> in your <b>detectObjectsFromVideo()</b> function. Once this is set, the extra parameter you sepecified in your function will be the Numpy array of the detected frame. See a sample below: <br><br>
<pre>
def forFrame(frame_number, output_array, output_count, detected_frame):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
	print("Returned Objects is : ", type(detected_frame))
    print("------------END OF A FRAME --------------")


video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          per_frame_function=forFrame,
                                          minimum_percentage_probability=40,
                                          log_progress=True, return_detected_frame=True)
</pre>


<div id="videodetectionintervals" ></div>
<h3><b><u>Frame Detection Intervals</u></b></h3>
The above video objects detection task are optimized for frame-real-time object detections that ensures that objects in every frame
of the video is detected. <b>ImageAI</b> provides you the option to adjust the video frame detections which can speed up
your video detection process. When calling the <b>.detectObjectsFromVideo()</b>, you can
specify at which frame interval detections should be made. By setting the <b>frame_detection_interval</b> parameter to be
 equal to 5 or 20, that means the object detections in the video will be updated after 5 frames or 20 frames.
If your output video <b>frames_per_second</b> is set to 20, that means the object detections in the video will
 be updated once in every quarter of a second or every second. This is useful in case scenarios where the available
 compute is less powerful and speeds of moving objects are low. This ensures you can have objects detected as second-real-time
, half-a-second-real-time or whichever way suits your needs. 
<br><br>

<div id="detectiontimeout"></div>
<h3><b><u>Custom Video Detection Timeout</u></b></h3>
<b>ImageAI</b> now allows you to set a timeout in seconds for detection of objects in videos or camera live feed. To set a timeout for your video detection code, all you need to do is specify the <b> detection_timeout </b> parameter in the <b>detectObjectsFromVideo()</b> function to the number of desired seconds. In the example code below, we set <b>detection_timeout</b> to 120 seconds (2 minutes). 
<br> <br>

<pre>
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()


camera = cv2.VideoCapture(0)

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,  minimum_percentage_probability=40,
                                          detection_timeout=120)
</pre>



<br>

<div id="documentation" ></div>
<h3><b><u> >> Documentation</u></b></h3>
We have provided full documentation for all <b>ImageAI</b> classes and functions in 3 major languages. Find links below: <br>

<b> >> Documentation - English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)</b> <br>
<b> >> Documentation - Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)</b>
<br>
<b> >> Documentation - French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)</b>

