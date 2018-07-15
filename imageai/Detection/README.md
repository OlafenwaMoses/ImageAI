# ImageAI : Object Detection <br>
<p>An <b>AI Commons</b> project <a href="https://commons.specpal.science" >https://commons.specpal.science </a></p>
<hr>
<br>
<h3><b><u>TABLE OF CONTENTS</u></b></h3>
<a href="#firstdetection" >&#9635 First Object Detection</a><br>
<a href="#objectextraction" >&#9635 Object Detection, Extraction and Fine-tune</a><br>
<a href="#customdetection" >&#9635 Custom Object Detection</a><br>
<a href="#detectionspeed" >&#9635 Detection Speed</a><br>
<a href="#hidingdetails" >&#9635 Hiding/Showing Object Name and Probability</a><br>
<a href="#inputoutputtype" >&#9635 Image Input & Output Types</a><br>
<a href="#support" >&#9635 Support the ImageAI Project</a><br>
<a href="#documentation" >&#9635 Documentation</a><br>
<br>
      ImageAI provides very convenient and powerful methods to perform object detection on images and extract
each object from the image. The object detection class supports RetinaNet, YOLOv3 and TinyYOLOv3. To start performing object detection,
you must download the RetinaNet, YOLOv3 or TinyYOLOv3 object detection model via the links below: <br> <br>
 <span><b>- <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5" style="text-decoration: none;" >RetinaNet</a></b> <b>(Size = 145 mb, high performance and accuracy, with longer detection time) </b></span> <br>

<span><b>- <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5" style="text-decoration: none;" >YOLOv3</a></b> <b>(Size = 237 mb, moderate performance and accuracy, with a moderate detection time) </b></span> <br>

<span><b>- <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5" style="text-decoration: none;" >TinyYOLOv3</a></b> <b>(Size = 34 mb, optimized for speed and moderate performance, with fast detection time) </b></span> <br><br>
 Once you download the object detection model file, you should copy the model file to the your project folder where your .py files will be.
 Then create a python file and give it a name; an example is FirstObjectDetection.py. Then write the code below into the python file: <br><br>

<div id="firstdetection" ></div>
 <h3><b>FirstObjectDetection.py</b></h3>

<b><pre>from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image2.jpg"), output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")


</pre></b><p>Sample Result:
    <br>
    <div style="width: 600px;" >
          <b><p><i>Input Image</i></p></b>
          <img src="../../images/image2.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br>
          <b><p><i>Output Image</i></p></b>
          <img src="../../images/yolo.jpg" style="width: 500px; height: auto; margin-left: 50px; " />
    </div> <br>
<pre>

laptop  :  87.32235431671143  :  (306, 238, 390, 284)
--------------------------------
laptop  :  96.86298966407776  :  (121, 209, 258, 293)
--------------------------------
laptop  :  98.6301600933075  :  (279, 321, 401, 425)
--------------------------------
laptop  :  99.78572130203247  :  (451, 204, 579, 285)
--------------------------------
bed  :  94.02391314506531  :  (23, 205, 708, 553)
--------------------------------
apple  :  48.03136885166168  :  (527, 343, 557, 364)
--------------------------------
cup  :  34.09906327724457  :  (462, 347, 496, 379)
--------------------------------
cup  :  44.65090036392212  :  (582, 342, 618, 386)
--------------------------------
person  :  57.70219564437866  :  (27, 311, 341, 437)
--------------------------------
person  :  85.26121377944946  :  (304, 173, 387, 253)
--------------------------------
person  :  96.33603692054749  :  (415, 130, 538, 266)
--------------------------------
person  :  96.95255160331726  :  (174, 108, 278, 269)
--------------------------------

</pre>

<br>
Let us make a breakdown of the object detection code that we used above.

<b><pre>
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
</pre></b>
 In the 3 lines above , we import the <b>ImageAI object detection </b> class in the first line, import the <b>os</b> in the second line and obtained
  the path to folder where our python file runs.
  <b><pre>
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
  </pre></b>
  In the 4 lines above, we created a new instance of the <b>ObjectDetection</b> class in the first line, set the model type to YOLOv3 in the second line,
  set the model path to the YOLOv3 model file we downloaded and copied to the python file folder in the third line and load the model in the
   fourth line.

   <b><pre>
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image2.jpg"), output_image_path=os.path.join(execution_path , "image2new.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")
</pre></b>

In the 2 lines above, we ran the <b>detectObjectsFromImage()</b> function and parse in the path to our image, and the path to the new
 image which the function will save. Then the function returns an array of dictionaries with each dictionary corresponding
 to the number of objects detected in the image. Each dictionary has the properties <b>name</b> (name of the object),
<b>percentage_probability</b> (percentage probability of the detection) and <b>box_points</b> ( the x1,y1,x2 and y2 coordinates of the bounding box of the object). <br>

Should you want to use the RetinaNet which is appropriate for high-performance and high-accuracy demanding detection tasks, you will download the RetinaNet model file from the links above, copy it to your python file's folder, set the model type and model path in your python code as seen below: <br>
<b><pre>
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
  </pre></b> <br>

However, if you desire TinyYOLOv3 which is optimized for speed and embedded devices, you will download the TinyYOLOv3 model file from the links above, copy it to your python file's folder, set the model type and model path in your python code as seen below: <br>
<b><pre>
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel()
  </pre></b>


<br><br>

<div id="objectextraction" ></div>
<h3><b><u> >> Object Detection, Extraction and Fine-tune</u></b></h3>

In the examples we used above, we ran the object detection on an image and it
returned the detected objects in an array as well as save a new image with rectangular markers drawn
 on each object. In our next examples, we will be able to extract each object from the input image
  and save it independently.
  <br>
  <br>
  In the example code below which is very identical to the previous object detction code, we will save each object
   detected as a seperate image.

   <b><pre>from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3new.jpg"), minimum_percentage_probability=30,  extract_detected_objects=True)

for eachObject, eachObjectPath in zip(detections, objects_path):
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("Object's image saved in " + eachObjectPath)
    print("--------------------------------")

</pre></b>

<br>
    <p>Sample Result:
    <br>
    <div style="width: 600px;" >
          <b><p><i>Input Image</i></p></b>
          <img src="../../images/image3.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br>
          <b><p><i>Output Images</i></p></b>
          <img src="../../images/image3new.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br> <div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/dog-1.jpg" style="width: 100px; height: auto; margin-left: 50px;  " /> <br>
            <center><i>dog</i></center>
          </div>
          <div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/motorcycle-3.jpg" style="width: 100px; height: auto; margin-left: 50px; " /> <br>
            <center><i>motorcycle</i></center>
          </div>
          <div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/car-4.jpg" style="width: 100px; height: auto; margin-left: 50px; " /> <br>
            <center><i>car</i></center>
          </div>
          <div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/bicycle-5.jpg" style="width: 100px; height: auto; margin-left: 50px;  " /> <br>
            <center><i>bicycle</i></center>
          </div>
          <div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/person-6.jpg" style="width: 100px; height: auto; margin-left: 50px;  " /> <br>
            <center><i>person</i></center>
          </div>
          <div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/person-7.jpg" style="width: 100px; height: auto; margin-left: 50px;  " /> <br>
            <center><i>person</i></center>
          </div>
          <div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/person-8.jpg" style="width: 100px; height: auto; margin-left: 50px;  " /> <br>
            <center><i>person</i></center>
          </div><div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/person-9.jpg" style="width: 100px; height: auto; margin-left: 50px;  " /> <br>
            <center><i>person</i></center>
          </div>
			<div style="width: 180px; margin-left: 10px;" >
            <img src="../../images/image3new.jpg-objects/person-10.jpg" style="width: 100px; height: auto; margin-left: 50px;  " /> <br>
            <center><i>person</i></center>
          </div>

    </div>

<br> <br>

Let us review the part of the code that perform the object detection and extract the images:

<b><pre>
detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3new.jpg"), minimum_percentage_probability=30,  extract_detected_objects=True)

for eachObject, eachObjectPath in zip(detections, objects_path):
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("Object's image saved in " + eachObjectPath)
    print("--------------------------------")
</pre></b>

In the above above lines, we called the <b>detectObjectsFromImage()</b> , parse in the input image path, output image part, and an
extra parameter <b>extract_detected_objects=True</b>. This parameter states that the function should extract each object detected from the image
and save it has a seperate image. The parameter is false by default. Once set to <b>true</b>, the function will create a directory
 which is the <b>output image path + "-objects"</b> . Then it saves all the extracted images into this new directory with
  each image's name being the <b>detected object name + "-" + a number</b> which corresponds to the order at which the objects
  were detected.
  <br><br>
This new parameter we set to extract and save detected objects as an image will make the function to return 2 values. The
 first is the array of dictionaries with each dictionary corresponding to a detected object. The second is an array of the paths
  to the saved images of each object detected and extracted, and they are arranged in order at which the objects are in the
  first array.

  <br><br>
  <b><h3>And one important feature you need to know!</h3></b> You will recall that the percentage probability
   for each detected object is sent back by the <b>detectObjectsFromImage()</b> function. The function has a parameter
   <b>minimum_percentage_probability</b> , whose default value is <b>50</b> (value ranges between 0 - 100) , but it set to 30 in this example. That means the function will only return a detected
    object if it's percentage probability is <b>30 or above</b>. The value was kept at this number to ensure the integrity of the
     detection results. You fine-tune the object
      detection by setting <b>minimum_percentage_probability</b> equal to a smaller value to detect more number of objects or higher value to detect less number of objects.

<br><br>

<div id="customdetection" ></div>
<h3><b><u>  >> Custom Object Detection</u></b></h3>
The object detection model (<b>RetinaNet</b>) supported by <b>ImageAI</b> can detect 80 different types of objects. They include: <br>
<pre>
      person,   bicycle,   car,   motorcycle,   airplane,
          bus,   train,   truck,   boat,   traffic light,   fire hydrant,   stop_sign,
          parking meter,   bench,   bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra,
          giraffe,   backpack,   umbrella,   handbag,   tie,   suitcase,   frisbee,   skis,   snowboard,
          sports ball,   kite,   baseball bat,   baseball glove,   skateboard,   surfboard,   tennis racket,
          bottle,   wine glass,   cup,   fork,   knife,   spoon,   bowl,   banana,   apple,   sandwich,   orange,
          broccoli,   carrot,   hot dog,   pizza,   donot,   cake,   chair,   couch,   potted plant,   bed,
          dining table,   toilet,   tv,   laptop,   mouse,   remote,   keyboard,   cell phone,   microwave,
          oven,   toaster,   sink,   refrigerator,   book,   clock,   vase,   scissors,   teddy bear,   hair dryer,
          toothbrush.
</pre>

Interestingly, <b>ImageAI</b> allow you to perform detection for one or more of the items above. That means you can
 customize the type of object(s) you want to be detected in the image. Let's take a look at the code below: <br>

<b><pre>from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(car=True, motorcycle=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3custom.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")

</pre></b>

<p>Result:
    <br>
    <div style="width: 600px;" >
          <img src="../../images/image3custom.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br>
    </div>
<br>

Let us take a look at the part of the code that made this possible.
<pre>custom_objects = detector.CustomObjects(car=True, motorcycle=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3custom.jpg"), minimum_percentage_probability=30)

</pre>
In the above code, after loading the model (can be done before loading the model as well), we defined a new variable
"<b>custom_objects = detector.CustomObjects()</b>", in which we set its car and motorcycle properties equal to <b>True</b>.
This is to tell the model to detect only the object we set to True. Then we call the "<b>detector.detectCustomObjectsFromImage()</b>"
which is the function that allows us to perform detection of custom objects. Then we will set the "<b>custom_objects</b>" value
 to the custom objects variable we defined.
<br><br>


<div id="detectionspeed"></div>
<h3><b><u>  >> Detection Speed</u></b></h3>
<b> ImageAI </b> now provides detection speeds for all object detection tasks. The detection speeds allow you to reduce
 the time of detection at a rate between 20% - 80%, and yet having just slight changes but accurate detection
results. Coupled with lowering the <b>minimum_percentage_probability</b> parameter, detections can match the normal
speed and yet reduce detection time drastically. The available detection speeds are <b>"normal"</b>(default), <b>"fast"</b>, <b>"faster"</b> , <b>"fastest"</b> and <b>"flash"</b>.
All you need to do is to state the speed mode you desire when loading the model as seen below.

<b><pre>detector.loadModel(detection_speed="fast")</pre></b> <br>

<br><br>

<div id="hidingdetails"></div>
<h3><b><u>  >> Hiding/Showing Object Name and Probability</u></b></h3>
<b>ImageAI</b> provides options to hide the name of objects detected and/or the percentage probability from being shown on the saved/returned detected image. Using the <b>detectObjectsFromImage()</b> and <b>detectCustomObjectsFromImage()</b> functions, the parameters <b>'display_object_name'</b> and <b>'display_percentage_probability'</b>  can be set to True of False individually. Take a look at the code below: <br>
<pre>
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3new_nodetails.jpg"), minimum_percentage_probability=30, display_percentage_probability=False, display_object_name=False)

</pre>

<br> In the above code, we specified that both the object name and percentage probability should not be shown. As you can see in the result below, both the names of the objects and their individual percentage probability is not shown in the detected image. <br>
<b><p><i>Result</i></p></b>
          <img src="../../images/nodetails.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br>


<br><br>


<div id="inputoutputtype"></div>
<h3><b><u>  >> Image Input & Output Types</u></b></h3>
<b>ImageAI</b> supports 3 input types of inputs which are <b>file path to image file</b>(default), <b>numpy array of image</b> and <b>image file stream</b>
as well as 2 types of output which are image <b>file</b>(default) and numpy  <b>array </b>.
This means you can now perform object detection in production applications such as on a web server and system
 that returns file in any of the above stated formats.
<br> To perform object detection with numpy array or file stream input, you just need to state the input type
in the <b>.detectObjectsFromImage()</b> function or the <b>.detectCustomObjectsFromImage()</b> function. See example below.

<pre>detections = detector.detectObjectsFromImage(input_type="array", input_image=image_array , output_image_path=os.path.join(execution_path , "image.jpg")) # For numpy array input type
detections = detector.detectObjectsFromImage(input_type="stream", input_image=image_stream , output_image_path=os.path.join(execution_path , "test2new.jpg")) # For file stream input type</pre><br> To perform object detection with numpy array output you just need to state the output type
in the <b>.detectObjectsFromImage()</b> function or the <b>.detectCustomObjectsFromImage()</b> function. See example below.

<pre>detected_image_array, detections = detector.detectObjectsFromImage(output_type="array", input_image="image.jpg" ) # For numpy array output type
</pre>

<br><br>

<div id="support"></div>
 <h3><b><u>Support the ImageAI Project</u></b></h3>
<img src="../../supportimage.jpg" style="width: 500px; height: auto; margin-left: 50px; " />
The <b>ImageAI</b> project is <b>free and open-source</b>. We are devoting lots of time and effort to provide industrial grade and the best of computer vision tools using state-of-the-art machine learning algorithms, in a way that amateur, intermediate and professional developers and researcher will find easy, convenient, independent and at no cost. We are asking the support of everyone who appreciates, uses and share in our dream for this project. Visit the link below to our <b>Indiegogo campaign</b> to contribute a token, or something substantial which will earn you an exclusive free E-Book that covers tutorials and full sample codes on using <b>ImageAI</b> for real-life and large-scale projects.
<br>
<b><h3> [ >>> Support ImageAI on Indiegogo]() </h3></b>
With your contributions, we will be adding more features including the ones requested by users of <b>ImageAI</b> that has contacted us. Some of the features are : <br> <br>
<b> 1) Custom training of Object Detection Models using RetinaNet, YOLOv3 and TinyYOLOv3</b> <br>
<b> 2) Image Segmentation</b> <br>
<b> 3) Face, Gender and Age Detection</b> <br>
<b> 4) Vehicle Number Plate Detection Recognition</b> <br>
<b> 5) ImageAI and all its features for Android</b> (For integrating all ImageAI features into Android Applications) <br>
<b> 6) ImageAI and all its features for iOS</b> (For integrating all ImageAI features into iOS Applications) <br>
<b> 7) ImageAI and all its features for .NET</b> (ImageAI and all its features for .NET developers) <br>

<div id="documentation" ></div>
<h3><b><u> >> Documentation</u></b></h3>
We have provided full documentation for all <b>ImageAI</b> classes and functions in 2 major languages. Find links below: <br>

<b> >> Documentation - English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)</b> <br>
<b> >> Documentation - Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)</b>
