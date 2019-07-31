# ImageAI : Custom Object Detection <br>
<p>An <b>DeepQuest AI</b> project <a href="https://deepquestai.com" >https://deepquestai.com </a></p>
<hr>
<br>
<h3><b><u>TABLE OF CONTENTS</u></b></h3>
<a href="#customdetection" >&#9635 Custom Object Detection</a><br>
<a href="#objectextraction" >&#9635 Object Detection, Extraction and Fine-tune</a><br>
<a href="#hidingdetails" >&#9635 Hiding/Showing Object Name and Probability</a><br>
<a href="#inputoutputtype" >&#9635 Image Input & Output Types</a><br>
<a href="#documentation" >&#9635 Documentation</a><br>
<br>
      ImageAI provides very convenient and powerful methods to perform object detection on images and extract
each object from the image using your own <b>custom YOLOv3 model</b> and the corresponding <b>detection_config.json</b> generated during the training. To test the custom object detection,
you can download a sample custom model we have trained to detect the Hololens headset and its <b>detection_config.json</b> file via the links below:<br> <br>
 <span><b>- <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/hololens-ex-60--loss-2.76.h5" style="text-decoration: none;" >hololens-ex-60--loss-2.76.h5</a></b> <b>(Size = 236 mb) </b></span> <br>

<span><b>- <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/detection_config.json" style="text-decoration: none;" >detection_config.json</a></b> <br>

<br><br>
 Once you download the custom object detection model file, you should copy the model file to the your project folder where your <b>.py</b> files will be.
 Then create a python file and give it a name; an example is FirstCustomDetection.py. Then write the code below into the python file: <br><br>

<div id="customdetection" ></div>
 <h3><b>FirstCustomDetection.py</b></h3>

<pre>from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("hololens-ex-60--loss-2.76.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="holo2.jpg", output_image_path="holo2-detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

</pre>
Sample Result - Input:
    
  <img src="../../../images/holo2.jpg" style="width: 600px; height: auto; " /> <br><br>
  Output: <br><br>
  <img src="../../../images/holo2-detected.jpg" style="width: 600px; height: auto; " /> <br>
          
<pre>
hololens  :  39.69653248786926  :  [611, 74, 751, 154]
hololens  :  87.6643180847168  :  [23, 46, 90, 79]
hololens  :  89.25175070762634  :  [191, 66, 243, 95]
hololens  :  64.49641585350037  :  [437, 81, 514, 133]
hololens  :  91.78624749183655  :  [380, 113, 423, 138]

</pre>

<br>
Let us make a breakdown of the object detection code that we used above.

<pre>
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
</pre>
 In the 3 lines above , we import the <b>ImageAI custom object detection </b> class in the first line, created the class instance on the second line and set the model type to YOLOv3.
  <pre>
detector.setModelPath("hololens-ex-60--loss-2.76.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()
  </pre>
  In the 3 lines above, we specified the file path to our downloaded model file in the first line , specified the path to our <b>detection_config.json</b> file in the second line and loaded the model on the third line.

   <pre>
detections = detector.detectObjectsFromImage(input_image="holo2.jpg", output_image_path="holo2-detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

</pre>

In the 3 lines above, we ran the <b>detectObjectsFromImage()</b> function and parse in the path to our test image, and the path to the new
 image which the function will save. Then the function returns an array of dictionaries with each dictionary corresponding
 to the number of objects detected in the image. Each dictionary has the properties <b>name</b> (name of the object),
<b>percentage_probability</b> (percentage probability of the detection) and <b>box_points</b> ( the x1,y1,x2 and y2 coordinates of the bounding box of the object). <br>


<br><br>

<div id="objectextraction" ></div>
<h3><b><u> Object Detection, Extraction and Fine-tune</u></b></h3>

In the examples we used above, we ran the object detection on an image and it
returned the detected objects in an array as well as save a new image with rectangular markers drawn
 on each object. In our next examples, we will be able to extract each object from the input image
  and save it independently.
  <br>
  <br>
  In the example code below which is very identical to the previous object detection code, we will save each object
   detected as a separate image.

   <pre>from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("hololens-ex-60--loss-2.76.h5")
detector.setJsonPath("detection_config.json") 
detector.loadModel()
detections, extracted_objects_array = detector.detectObjectsFromImage(input_image="holo2.jpg", output_image_path="holo2-detected.jpg", extract_detected_objects=True)

for detection, object_path in zip(detections, extracted_objects_array):
    print(object_path)
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("---------------")

</pre>

<br>
    <p>Sample Result:
    <br>
       
          <b><p><i>Output Images</i></p></b>
          
            <img src="../../../images/holo2-detected.jpg-objects/hololens-1.jpg" style="width: 200px; height: auto; margin-left: 50px;  " /> <br>
            
            <img src="../../../images/holo2-detected.jpg-objects/hololens-2.jpg" style="width: 200px; height: auto; margin-left: 50px;  " /> <br>
            
            <img src="../../../images/holo2-detected.jpg-objects/hololens-3.jpg" style="width: 200px; height: auto; margin-left: 50px;  " /> <br>
            <img src="../../../images/holo2-detected.jpg-objects/hololens-4.jpg" style="width: 200px; height: auto; margin-left: 50px;  " /> <br>
            
            <img src="../../../images/holo2-detected.jpg-objects/hololens-5.jpg" style="width: 200px; height: auto; margin-left: 50px;  " /> <br>
            
            <img src="../../../images/holo2-detected.jpg-objects/hololens-6.jpg" style="width: 200px; height: auto; margin-left: 50px;  " /> <br>
            <img src="../../../images/holo2-detected.jpg-objects/hololens-7.jpg" style="width: 200px; height: auto; margin-left: 50px;  " /> <br>
            
          

    

<br> <br>

Let us review the part of the code that perform the object detection and extract the images:

<pre>
detections, extracted_objects_array = detector.detectObjectsFromImage(input_image="holo2.jpg", output_image_path="holo2-detected.jpg", extract_detected_objects=True)

for detection, object_path in zip(detections, extracted_objects_array):
    print(object_path)
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("---------------")
</pre>

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
   <b>minimum_percentage_probability</b> , whose default value is <b>30</b> (value ranges between 0 - 100) , but it set to 30 in this example. That means the function will only return a detected
    object if it's percentage probability is <b>30 or above</b>. The value was kept at this number to ensure the integrity of the
     detection results. You fine-tune the object
      detection by setting <b>minimum_percentage_probability</b> equal to a smaller value to detect more number of objects or higher value to detect less number of objects.

<br><br>


<div id="hidingdetails"></div>
<h3><b><u> Hiding/Showing Object Name and Probability</u></b></h3>
<b>ImageAI</b> provides options to hide the name of objects detected and/or the percentage probability from being shown on the saved/returned detected image. Using the <b>detectObjectsFromImage()</b> and <b>detectCustomObjectsFromImage()</b> functions, the parameters <b>'display_object_name'</b> and <b>'display_percentage_probability'</b>  can be set to True of False individually. Take a look at the code below: <br>
<pre>
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "holo2.jpg"), output_image_path=os.path.join(execution_path , "holo2_nodetails.jpg"), minimum_percentage_probability=30, display_percentage_probability=False, display_object_name=False)

</pre>

<br> In the above code, we specified that both the object name and percentage probability should not be shown. As you can see in the result below, both the names of the objects and their individual percentage probability is not shown in the detected image. <br>
<b><p><i>Result</i></p></b>
          <img src="../../../images/holo2-nodetails.jpg" style="width: 600px; height: auto; " /> <br>


<br><br>


<div id="inputoutputtype"></div>
<h3><b><u>Image Input & Output Types</u></b></h3>
<b>ImageAI</b> custom object detection supports 2 input types of inputs which are <b>file path to image file</b>(default) and <b>numpy array of an image</b>
as well as 2 types of output which are image <b>file</b>(default) and numpy <b>array </b>.
This means you can now perform object detection in production applications such as on a web server and system
 that returns file in any of the above stated formats.
<br> To perform object detection with numpy array input, you just need to state the input type
in the <b>.detectObjectsFromImage()</b> function. See example below.

<pre>detections = detector.detectObjectsFromImage(input_type="array", input_image=image_array , output_image_path=os.path.join(execution_path , "holo2-detected.jpg")) # For numpy array input type
</pre><br> To perform object detection with numpy array output you just need to state the output type
in the <b>.detectObjectsFromImage()</b> function. See example below.

<pre>detected_image_array, detections = detector.detectObjectsFromImage(output_type="array", input_image="holo2.jpg" ) # For numpy array output type
</pre>

<br><br>

<div id="documentation" ></div>
<h3><b><u> >> Documentation</u></b></h3>
We have provided full documentation for all <b>ImageAI</b> classes and functions in 3 major languages. Find links below: <br>

<b> >> Documentation - English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)</b> <br>
<b> >> Documentation - Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)</b>
<br>
<b> >> Documentation - French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)</b>
