# ImageAI
A python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities
<hr>
Built with simplicity in mind, ImageAI 
    supports a list of state-of-the-art Machine Learning algorithms for image recognition.
    ImageAI currently supports image recognition using 4 different Machine Learning algorithms 
    trained on the ImageNet-1000 dataset, meaning any application built with <b>ImageAI</b>
    can recognize 1000 distinct objects including vehicles, animals, plants, places, electronics,
    indoor objects, outdoor objects and more. <br>
                                   Eventually, <b>ImageAI</b> will provide support for a wider
    and more specialized aspects of Computer Vision including and not limited to image 
    recognition in special environments and special fields, object detection and custom image
    prediction.
</p>
<br><br>

<h3><b><u>Dependencies</u></b></h3>
                            To use <b>ImageAI</b> in your application developments, you must have installed the following 
 dependencies before you install <b>ImageAI</b> : <br> <br>
       <span><b>- Python 3.5.1 (and later versions) </b>      <a href="https://www.python.org/downloads/" style="text-decoration: none;" >Download<a> (Support for Python 2.7 coming soon) </span> <br> 
       <span><b>- pip3 </b>              <a href="https://pypi.python.org/pypi/pip" style="text-decoration: none;" >Install<a></span> <br>
       <span><b>- Tensorflow 1.4.0 (and later versions)  </b>      <a href="https://www.tensorflow.org/install/install_windows" style="text-decoration: none;" > Install<a></span> <br>
       <span><b>- Numpy 1.13.1 (and later versions) </b>      <a href="https://www.scipy.org/install.html" style="text-decoration: none;" >Install<a></span> <br>
       <span><b>- SciPy 0.19.1 (and later versions) </b>      <a href="https://www.scipy.org/install.html" style="text-decoration: none;" >Install<a></span> <br>

 <h3><b><u>Installation</u></b></h3>
      To install ImageAI, download the Python Wheel <a href="dist/imageai-1.0.1-py3-none-any.whl" ><b>
    imageai-1.0.1-py3-none-any.whl</b></a> and run the python installation instruction in the Windows command line
     to the path of the file like the one below: <br><br>
    <span>      <b>pip3 install C:\User\MyUser\Downloads\imageai-1.0.1-py3-none-any.whl</b></span> <br><br>


<h3><b><u>Using ImageAI</u></b></h3>
      ImageAI provides 4 different algorithms and model types to perform image prediction.
To perform image prediction on any picture, take the following simple steps.  The 4 algorithms provided for
 image prediction include <b>SqueezeNet</b>, <b>ResNet</b>, <b>InceptionV3</b> and <b>DenseNet</b>. Each of these
  algorithms have individual model files which you must use depending on the choice of your algorithm. To download the
   model file for your choice of algorithm, click on any of the links below: <br> <br>
       <span><b>- <a href="https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5" style="text-decoration: none;" >SqueezeNet</a> (Size = 4.82 mb, fastest prediction time and moderate accuracy) </b></span> <br>
       <span><b>- <a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5" style="text-decoration: none;" >ResNet</a></b> by Microsoft Research <b>(Size = 98 mb, fast prediction time and high accuracy) </b></span> <br>
       <span><b>- <a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5" style="text-decoration: none;" >InceptionV3</a></b> by Google Brain team <b>(Size = 91.6 mb, slow prediction time and higher accuracy) </b></span> <br>
       <span><b>- <a href="https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32.h5" style="text-decoration: none;" >DenseNet</a></b> by Facebook AI Research <b>(Size = 31.6 mb, slower prediction time and highest accuracy) </b></span> <br><br>

   After you download the model file of your choice, you will need to download one more file which will be a JSON file that contains the model
    mapping for all the 1000 objects supported. Find the link below: <br> <br>
        <span><b>- <a href="imagenet_class_index.json" download style="text-decoration: none;" >imagenet_class_index.json</a> </b></span> <br> <br>
       Great! Once you have downloaded these two files, start a new python project, and then copy these two files to your project
     folder where your python files (.py files) will be . Then create a python file and give it a name; an example is <b>FirstPrediction.py</b>.
      Then write the code below into the python file: <br><br>

<h3><b>FirstPrediction.py</b></h3>
<b><pre>
from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()
prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath( execution_path + "\\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.setJsonPath(execution_path + "\\imagenet_class_index.json")
prediction.loadModel()

predictions, percentage_probabilities = prediction.predictImage("C:\\Users\\MyUser\\Downloads\\sample.jpg", result_count=5)
for index in range(len(predictions)):
        print(predictions[index] + " : " + percentage_probabilities[index])

</pre></b>

<p>Sample Result:
    <br>
    <img src="sample.jpg" style="width: 400px; height: auto;" /> 
    <pre>sports_car : 90.61029553413391
car_wheel : 5.9294357895851135
racer : 0.9972884319722652
convertible : 0.8457873947918415
grille : 0.581052340567112</pre>
</p>

<br>
<span>
          The code above works as follows: <br>
     <b><pre>from imageai.Prediction import ImagePrediction
import os</pre></b>
<br>
      The code above imports the <b>ImageAI</b> library 
 and the python <b>os</b> class. <br>
<b><pre>execution_path = os.getcwd()</pre></b>
<br> The above line obtains the path to the folder that contains
your python file (in this example, your FirstPrediction.py) . <br>

<b><pre>prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(execution_path + "\\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.setJsonPath(execution_path + "\\imagenet_class_index.json")</pre></b>
      In the lines above, we created and instance of the <b>ImagePrediction()</b>
 class in the first line, then we set the model type of the prediction object to ResNet by caling the <b>.setModelTypeAsResNet()</b>
  in the second line, then we set the model path of the prediction object to the path of the model file (<b>resnet50_weights_tf_dim_ordering_tf_kernels.h5</b>) we copied to the python file folder
  in the third line, and then set the json path of the prediction object to the path of the json file (<b>imagenet_class_index.json</b>) you copied to the python file folder.

<b><pre>predictions, percentage_probabilities = prediction.predictImage("C:\\Users\\MyUser\\Downloads\\sample.jpg", result_count=5)</pre></b> In the above line, we defined 2 variables to be equal to the function
 called to predict an image, which is the <b>.predictImage()</b> function, into which we parsed the path to 
 our image and also state the number of prediction results we want to have (values from 1 to 1000) parsing 
 <b> result_count=5 </b>. The <b>.predictImage()</b> function will return 2 array objects with the first (<b>predictions</b>) being
  an array of predictions and the second (<b>percentage_probabilities</b>) being an array of the corresponding percentage probability for each 
  prediction.

  <b><pre>for index in range(len(predictions)):
        print(predictions[index] + " : " + percentage_probabilities[index])</pre></b> The above line obtains each object in the <b>predictions</b> array, and also 
obtain the corresponding percentage probability from the <b>percentage_probabilities</b>, and finally prints
the result of both to console.

</span>


<br><br>

<h3><b><u>Using ImageAI in MultiThreading</u></b></h3>
       When developing programs that run heavy task on the deafult thread like User Interfaces (UI),
 you should consider running your predictions in a new thread. When running image prediction using ImageAI in 
 a new thread, you must take note the following: <br>
         - You can create your prediction object, set its model type, set model path and json path
outside the new thread. <br>
          - The <b>.loadModel()</b> must be in the new thread and image prediction (<b>predictImage()</b>) must take place in th new thread.
<br>
      Take a look of a sample code below on image prediction using multithreading:
<pre><b>
from imageai.Prediction import ImagePrediction
import os
import threading

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath( execution_path + "\\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.setJsonPath(execution_path + "\\imagenet_class_index.json")

picturesfolder = os.environ["USERPROFILE"] + "\\Pictures\\"
allfiles = os.listdir(picturesfolder)

class PredictionThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        prediction.loadModel()
        for eachPicture in allfiles:
            if eachPicture.endswith(".png") or eachPicture.endswith(".jpg"):
                predictions, percentage_probabilities = prediction.predictImage(picturesfolder + eachPicture, result_count=1)
                for index in range(len(predictions)):
                    print(predictions[index] + " : " + percentage_probabilities[index])

predictionThread = PredictionThread ()
predictionThread.start()
    </b></pre>

<br>

<h3><b><u>Sample Application</u></b></h3>
      As a demonstration of the what you can do with ImageAI, we have 
 built a complete AI powered Photo gallery for Windows called <b>IntelliP</b> ,  using <b>ImageAI</b> and UI framework <b>Kivy</b>. Follow this 
 <a href=""  > link </a> to download page of the application and its source code.
<br>

<h3><b><u>Documentation</u></b></h3>
 <span>       The ImageAI library currently supports only image prediction via the <b>ImagePrediction</b>
  class. 

 </span>
 <br>  <br>
 <span style="font-size: 20px;" >The <b>ImagePrediction</b>  class </span> 
 <hr>
 <p>
           The <b>ImagePrediction</b> class can be used to perform image prediction
      in any python application by initiating it and calling the available functions below: <br>
            <b>- setModelTypeAsSqueezeNet()</b>    This function should be called should you 
      chose to use the SqueezeNet model file for the image prediction. You only need to call it once. <br>
            <b>- setModelTypeAsResNet()</b>    This function should be called should you 
      chose to use the ResNet model file for the image prediction. You only need to call it once. <br>
            <b>- setModelTypeAsInceptionV3()</b>    This function should be called should you 
      chose to use the InceptionV3Net model file for the image prediction. You only need to call it once. <br>
            <b>- setModelTypeAsDenseNet</b>    This function should be called should you 
      chose to use the DenseNet model file for the image prediction. You only need to call it once. <br>
            <b>- setModelPath()</b>    You need to call this function only once and parse the path to
       the model file path into it. The model file type must correspond to the model type you set.  <br>
            <b>- setJsonPath()</b>    You need to call this function only once and parse the path to
       the <b>imagenet_class_index.json</b> into it. <br>
             <b>- loadModel()</b>    You need to call this function
        once only before you attempt to call the <b>predictImage()</b> function .<br>
              <b>- predictImage()</b>    To perform image 
         prediction on an image, you will call this function and parse in the path to the image file
          you want to predict, and also state the number of predictions result you will want to 
           have returned by the function (1 to 1000 posible results). This functions returns two arrays.
           The first one is an array of predictions while the second is an array of corresponding percentage probabilities 
           for each prediction in the prediction array. You can call this function as many times as you need 
           for as many images you want to predict and at any point in your python program as far as you
            have called the required functions to set model type, set model file path, set json file path
             and load the model.  <br>
 </p>
 

 <h3><b><u>Contact Developers</u></b></h3>
 <p> <b>Moses Olafenwa</b> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://moses.specpal.science"> https://moses.specpal.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/OlafenwaMoses"> @OlafenwaMoses</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@guymodscientist"> @guymodscientist</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" target="_blank" href="https://facebook.com/moses.olafenwa"> moses.olafenwa</a> <br>
<br><br>
      <b>John Olafenwa</b> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://john.specpal.science"> https://john.specpal.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/johnolafenwa"> @johnolafenwa</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@johnolafenwa"> @johnolafenwa</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" href="https://facebook.com/olafenwajohn"> olafenwajohn</a> <br>


 </p>
