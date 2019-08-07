# ImageAI (v2.1.2 )<br>
A python library built to empower developers to build applications and systems with self-contained Deep Learning and Computer Vision capabilities using simple
 and few lines of code. <br><br>

<img src="logo1.png" style="width: 500px; height: auto; margin-left: 50px; " /> <br>

An <b>DeepQuest AI</b> project <a href="https://deepquestai.com" >https://deepquestai.com </a>.
Developed and Maintained by [Moses Olafenwa](https://twitter.com/OlafenwaMoses) and [John Olafenwa](https://twitter.com/johnolafenwa), brothers, creators of [TorchFusion](https://github.com/johnolafenwa/TorchFusion), Authors of [Introduction to Deep Computer Vision](https://john.aicommons.science/deepvision) and creators of [DeepStack AI Server](https://deepstack.cc).
<hr>

Built with simplicity in mind, <b>ImageAI</b> 
    supports a list of state-of-the-art Machine Learning algorithms for image prediction, custom image prediction, object detection, video detection, video object tracking
    and image predictions trainings. <b>ImageAI</b> currently supports image prediction and training using 4 different Machine Learning algorithms 
    trained on the ImageNet-1000 dataset. <b>ImageAI</b> also supports object detection, video detection and object tracking  using RetinaNet, YOLOv3 and TinyYOLOv3 trained on COCO dataset. Finally, <b>ImageAI</b> allows you to train custom models for performing detection and recognition of new objects. <br>
                                   Eventually, <b>ImageAI</b> will provide support for a wider
    and more specialized aspects of Computer Vision including and not limited to image 
    recognition in special environments and special fields.

<br> 

<b>New Release : ImageAI 2.1.0</b>
<br> What's new:
<br>
- Training of <b>custom YOLOv3</b> models on custom image datasets annotated in Pascal VOC format <br>
- Single and multi-model <b>mAP evaluation</b> of saved custom detection models<br>  
- Object detection in images using <b>custom YOLOv3 models</b><br> 
- Video Object detection in using <b>custom YOLOv3 models</b><br> 
- Video Detection analysis in using <b>custom YOLOv3 models</b>
- Support for <b>file</b> and <b>Numpy array</b> inputs/outputs for all custom image and video detections</b>



 <br>  <br>
<br>

<h3><b><u>TABLE OF CONTENTS</u></b></h3>
<a href="#dependencies" > &#9635 Dependencies</a><br>
<a href="#installation" > &#9635 Installation</a><br>
<a href="#prediction" > &#9635 Image Prediction</a><br>
<a href="#detection" > &#9635 Object Detection</a><br>
<a href="#videodetection" > &#9635 Video Object Detection, Tracking & Analysis</a><br>
 <a href="#customtraining" >&#9635 Custom Model Training</a><br>
<a href="#customprediction" > &#9635 Custom Image Prediction</a><br>
<a href="#customdetectiontraining" >&#9635 Custom Detection Model Training</a><br>
<a href="#customdetection" > &#9635 Custom Object Detection</a><br>
<a href="#customvideodetection" > &#9635 Custom Video Object Detection & Analysis</a><br>
<a href="#documentation" > &#9635 Documentation</a><br>
<a href="#sample" > &#9635 Projects Built on ImageAI</a><br>
<a href="#real-time-and-high-performance-implementation" >&#9635 High Performance Implementation</a><br>
<a href="#recommendation" >&#9635 AI Practice Recommendations</a><br>
<a href="#contact" >&#9635 Contact Developers</a><br>
<a href="#contributors" >&#9635 Contributors</a><br>
<a href="#ref" >&#9635 References</a><br>


<br><br>

<div id="dependencies"></div>
<h3><b><u>Dependencies</u></b></h3>To use <b>ImageAI</b> in your application developments, you must have installed the following 
 dependencies before you install <b>ImageAI</b> : 
 
 
 <br> <br>
       <span><b>- Python 3.5.1 (and later versions) </b>      <a href="https://www.python.org/downloads/" style="text-decoration: none;" >Download</a> (Support for Python 2.7 coming soon) </span> <br>
       <span><b>- pip3 </b>              <a href="https://pypi.python.org/pypi/pip" style="text-decoration: none;" >Install</a></span> <br>
       <span><b>- Tensorflow 1.4.0 (and later versions)  </b>      <a href="https://www.tensorflow.org/install/install_windows" style="text-decoration: none;" > Install</a></span> or install via pip <pre> pip3 install --upgrade tensorflow </pre> 
       
  <span><b>- OpenCV  </b>        <a href="https://pypi.python.org/pypi/opencv-python" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install opencv-python </pre> 
       
   <span><b>- Keras 2.x  </b>     <a href="https://keras.io/#installation" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install keras </pre> 

<div id="installation"></div>
 <h3><b><u>Installation</u></b></h3> To install ImageAI, run the python installation instruction below in the command line: <br><br>
    <span>      <b><pre>pip3 install imageai --upgrade </pre></b></span> <br><br> <br>
    
   

<div id="prediction"></div>
<h3><b><u>Image Prediction</u></b></h3>
<p><img src="images/1.jpg" style="width: 400px; height: auto;" /> 
    <pre>convertible : 52.459555864334106
sports_car : 37.61284649372101
pickup : 3.1751200556755066
car_wheel : 1.817505806684494
minivan : 1.7487050965428352</pre>
</p>

<b>ImageAI</b> provides 4 different algorithms and model types to perform image prediction, trained on the ImageNet-1000 dataset.
The 4 algorithms provided for image prediction include <b>SqueezeNet</b>, <b>ResNet</b>, <b>InceptionV3</b> and <b>DenseNet</b>. 

Click the link below to see the full sample codes, explanations and best practices guide.

<a href="imageai/Prediction/" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorial & Guide </button></a>

<br>

<br>


<div id="detection"></div>
<h3><b><u>Object Detection</u></b></h3>
<div style="width: 600px;" >
          <b><p><i>Input Image</i></p></b></br>
          <img src="images/image2.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br>
          <b><p><i>Output Image</i></p></b>
          <img src="images/image2new.jpg" style="width: 500px; height: auto; margin-left: 50px; " />
    </div> <br>
<pre>

person : 91.946941614151
--------------------------------
person : 73.61021637916565
--------------------------------
laptop : 90.24320840835571
--------------------------------
laptop : 73.6881673336029
--------------------------------
laptop : 95.16398310661316
--------------------------------
person : 87.10319399833679
--------------------------------

</pre>
<b>ImageAI</b> provides very convenient and powerful methods
 to perform object detection on images and extract each object from the image. The object detection class provides support for RetinaNet, YOLOv3 and TinyYOLOv3, with options to adjust for state of the art performance or real time processing.

Click the link below to see the full sample codes, explanations and best practices guide.


<a href="imageai/Detection/" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorial & Guide</button></a>

<br><br>







<div id="videodetection"></div>
<h3><b><u>Video Object Detection and Tracking</u></b></h3>

<p><div style="width: 600px;" >
          <p><i><b>Video Object Detection & Analysis</b></i></p>
<p><i>Below is a snapshot of a video with objects detected.</i></p>
          <img src="images/video1.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br>
          <p><i><b>Video Custom Object Detection (Object Tracking)</b></i></p>
            <p><i>Below is a snapshot of a video with only person, bicycle and motorcyle detected.</i></p>
          <img src="images/video2.jpg" style="width: 500px; height: auto; margin-left: 50px; " /><br>
          <p><i><b>Video Analysis Visualization </b></i></p>
            <p><i>Below is a visualization of video analysis returned by **ImageAI** into a 'per_second' function.</i></p>
          <img src="images/video_analysis_visualization.jpg" style="width: 500px; height: auto; margin-left: 50px; " />
    </div> <br>

<b>ImageAI</b> provides very convenient and powerful methods
 to perform object detection in videos and track specific object(s). The video object detection class provided only supports
 the current state-of-the-art RetinaNet, but with options to adjust for state of the art performance or real time processing.
Click the link to see the full videos, sample codes, explanations and best practices guide.


</p>


<a href="imageai/Detection/VIDEO.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorial & Guide </button></a>

<br>

<div id="customtraining"></div>
<h3><b><u>Custom Model Training </u></b></h3>

<p><br>
    <div style="width: 600px;" >
            <p><i>A sample from the IdenProf Dataset used to train a Model for predicting professionals.</i></p>
          <img src="images/idenprof.jpg" style="width: 500px; height: auto; margin-left: 50px; " />
    </div> <br>


</p>

<b>ImageAI</b> provides classes and methods for you to train a new model that can be used to perform prediction on your own custom objects.
You can train your custom models using SqueezeNet, ResNet50, InceptionV3 and DenseNet in  <b> 5 </b> lines of code.
Click the link below to see the guide to preparing training images, sample training codes, explanations and best practices.
<br>


<a href="imageai/Prediction/CUSTOMTRAINING.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorials & Documentation </button></a>



<br><br>

<div id="customprediction"></div>
<h3><b><u>Custom Image Prediction </u></b></h3>

<p>
<p><i>Prediction from a sample model trained on IdenProf, for predicting professionals</i></p>
      <img src="images/4.jpg" style="width: 400px; height: auto;" />
    <pre>mechanic : 76.82620286941528
chef : 10.106072574853897
waiter : 4.036874696612358
police : 2.6663416996598244
pilot : 2.239348366856575</pre>


</p>

<b>ImageAI</b> provides classes and methods for you to run image prediction your own custom objects using your own model trained with <b>ImageAI</b> Model Training class.
You can use your custom models trained with SqueezeNet, ResNet50, InceptionV3 and DenseNet and the JSON file containing the mapping of the custom object names.
Click the link below to see the guide to sample training codes, explanations, and best practices guide.
<br>


<a href="imageai/Prediction/CUSTOMPREDICTION.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorials & Documentation </button></a>



<br><br>

<div id="customdetectiontraining"></div>
<h3><b><u>Custom Detection Model Training </u></b></h3>

<p>
<p><i>Training detection models to detect and recognize new objects.  </i></p>
      <img src="images/headsets.jpg" style="width: 600px; height: auto;" />
</p>

<b>ImageAI</b> provides classes and methods for you to train new <b>YOLOv3</b> object detection models on your <b>custom dataset</b>.
This means you can train a model to detect literally any object of interest by providing the images, the annotations and training with ImageAI.
Click the link below to see the guide to sample training codes, explanations, and best practices guide.
<br>


<a href="imageai/Detection/Custom/CUSTOMDETECTIONTRAINING.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorials & Documentation </button></a>

<br><br>

<div id="customdetection"></div>
<h3><b><u>Custom Object Detection </u></b></h3>

<p>
<p><i>Detection result from a custom YOLOv3 model trained to detect the Hololens headset.</i></p>
      <img src="images/holo2-detected.jpg" style="width: 600px; height: auto;" />
    <pre>hololens  :  39.69653248786926  :  [611, 74, 751, 154]
hololens  :  87.6643180847168  :  [23, 46, 90, 79]
hololens  :  89.25175070762634  :  [191, 66, 243, 95]
hololens  :  64.49641585350037  :  [437, 81, 514, 133]
hololens  :  91.78624749183655  :  [380, 113, 423, 138]</pre>


</p>

<b>ImageAI</b> now provides classes and methods for you detect and recognize your own custom objects in images using your own model trained with the <b>DetectionModelTraining</b> class.
You can use your custom trained <b>YOLOv3</b> mode and the <b>detection_config.json</b> file generated during the training.
Click the link below to see the guide to sample training codes, explanations, and best practices guide.
<br>


<a href="imageai/Detection/Custom/CUSTOMDETECTION.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorials & Documentation </button></a>


<br><br>

<div id="customvideodetection"></div>
<h3><b><u>Custom Video Object Detection & Analysis </u></b></h3>

<p>
<p><i>Video Detection result from a custom YOLOv3 model trained to detect the Hololens headset in a video.</i></p>
      <img src="images/customvideodetection.gif" style="width: 700px; height: auto;" />
      
    


</p>

Now you can use your custom trained <b>YOLOv3</b> model to detect, recognize and analyze objects in videos.
Click the link below to see the guide to sample training codes, explanations, and best practices guide.
<br>


<a href="imageai/Detection/Custom/CUSTOMVIDEODETECTION.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorials & Documentation </button></a>


<br><br>
<div id="documentation"></div>
<h3><b><u>Documentation </u></b></h3>

We have provided full documentation for all <b>ImageAI</b> classes and functions in 2 major languages. Find links below: <br>

<b> >> Documentation - English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)</b> <br>
<b> >> Documentation - Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)</b> <br>
<b> >> Documentation - French Version  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)</b>

<br> <br>

<div id="performance"></div>
<h3><b><u>Real-Time and High Performance Implementation</u></b></h3>
<b>ImageAI</b> provides abstracted and convenient implementations of state-of-the-art Computer Vision technologies. All of <b>ImageAI</b> implementations and code can work on any computer system with moderate CPU capacity. However, the speed of processing for operations like image prediction, object detection and others on CPU is slow and not suitable for real-time applications. To perform real-time Computer Vision operations with high performance, you need to use GPU enabled technologies.
<br> <br>
<b>ImageAI</b> uses the Tensorflow backbone for it's Computer Vision operations. Tensorflow supports both CPUs and GPUs ( Specifically NVIDIA GPUs.  You can get one for your PC or get a PC that has one) for machine learning and artificial intelligence algorithms' implementations. To use Tensorflow that supports the use of GPUs, follow the link below :
<br> <br>
FOR WINDOWS <br>
<a href="https://www.tensorflow.org/install/install_windows" >https://www.tensorflow.org/install/install_windows</a> <br><br>


FOR macOS <br>
<a href="https://www.tensorflow.org/install/install_mac" >https://www.tensorflow.org/install/install_mac</a> <br><br>


FOR UBUNTU <br>
<a href="https://www.tensorflow.org/install/install_linux">https://www.tensorflow.org/install/install_linux</a>
<br><br>

<div id="sample"></div>
<h3><b><u>Projects Built on ImageAI</u></b></h3> As a demonstration of  what you can do with ImageAI, we have 
 built a complete AI powered Photo gallery for Windows called <b>IntelliP</b> ,  using <b>ImageAI</b> and UI framework <b>Kivy</b>. Follow this 
 <a href="https://github.com/OlafenwaMoses/IntelliP"  > link </a> to download page of the application and its source code. <br> <br>

 We also welcome submissions of applications and systems built by you and powered by ImageAI for listings here. Should you want your ImageAI powered 
  developments listed here, you can reach to us via our <a href="#contact" >Contacts</a> below.

 <br> <br>

<div id="recommendation"></div>
 <h3><b><u>AI Practice Recommendations</u></b></h3>

 For anyone interested in building AI systems and using them for business, economic,  social and research purposes, it is critical that the person knows the likely positive, negative and unprecedented impacts the use of such technologies will have. They must also be aware of approaches and practices recommended by experienced industry experts to ensure every use of AI brings overall benefit to mankind. We therefore recommend to everyone that wishes to use ImageAI and other AI tools and resources to read Microsoft's January 2018 publication on AI titled "The Future Computed : Artificial Intelligence and its role in society ".
Kindly follow the link below to download the publication.
 <br><br>
<a href="https://blogs.microsoft.com/blog/2018/01/17/future-computed-artificial-intelligence-role-society/" >https://blogs.microsoft.com/blog/2018/01/17/future-computed-artificial-intelligence-role-society/</a>
 <br> <br>





<div id="contact"></div>
 <h3><b><u>Contact Developers</u></b></h3>
 <p> <b>Moses Olafenwa</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:guymodscientist@gmail.com"> guymodscientist@gmail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://moses.aicommons.science"> https://moses.aicommons.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/OlafenwaMoses"> @OlafenwaMoses</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@guymodscientist"> @guymodscientist</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" target="_blank" href="https://facebook.com/moses.olafenwa"> moses.olafenwa</a> <br>
<br><br>
      <b>John Olafenwa</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:johnolafenwa@gmail.com"> johnolafenwa@gmail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://john.aicommons.science"> https://john.aicommons.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/johnolafenwa"> @johnolafenwa</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@johnolafenwa"> @johnolafenwa</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" href="https://facebook.com/olafenwajohn"> olafenwajohn</a> <br>


 </p>

 <br><br>

<div id="contact"></div>
 <h3><b><u>Contributors</u></b></h3>
We are inviting anyone who wishes to contribute to the <b>ImageAI</b> project to reach to us. We primarily need contributions in translating the documentation of the project's code to major languages that includes but not limited to French, Spanish, Portuguese, Arabian and more. We want every developer and researcher around the world to benefit from this project irrespective of their native languages. <br>

We give special thanks to <b>[Kang vcar](https://github.com/kangvcar/)</b> for his incredible and excellent work in translating <b>ImageAI</b>'s documentation to the Chinese language. Find below the contact details of those who have contributed immensely to the <b>ImageAI</b> project.

 <p> <b>Kang vcar</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:kangvcar@mail.com"> kangvcar@mail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="http://www.kangvcar.com"> http://www.kangvcar.com</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/kangvcar"> @kangvcar</a> <br>



 </p>

 <br>

 <div id="ref"></div>
 <h3><b><u>References</u></b></h3>

 1. Somshubra Majumdar, DenseNet Implementation of the paper, Densely Connected Convolutional Networks in Keras <br>
 <a href="https://github.com/titu1994/DenseNet/" >https://github.com/titu1994/DenseNet/</a> <br>

 2. Broad Institute of MIT and Harvard, Keras package for deep residual networks <br>
 <a href="https://github.com/broadinstitute/keras-resnet" >https://github.com/broadinstitute/keras-resnet</a> <br>

 3. Fizyr, Keras implementation of RetinaNet object detection <br>
 <a href="https://github.com/fizyr/keras-retinanet" >https://github.com/fizyr/keras-retinanet</a> <br>

 4. Francois Chollet, Keras code and weights files for popular deeplearning models <br>
 <a href="https://github.com/fchollet/deep-learning-models" >https://github.com/fchollet/deep-learning-models</a> <br>

 5. Forrest N. et al, SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size <br>
 <a href="https://arxiv.org/abs/1602.07360" >https://arxiv.org/abs/1602.07360</a> <br>

 6. Kaiming H. et al, Deep Residual Learning for Image Recognition <br>
 <a href="https://arxiv.org/abs/1512.03385" >https://arxiv.org/abs/1512.03385</a> <br>

 7. Szegedy. et al, Rethinking the Inception Architecture for Computer Vision <br>
 <a href="https://arxiv.org/abs/1512.00567" >https://arxiv.org/abs/1512.00567</a> <br>

 8. Gao. et al, Densely Connected Convolutional Networks <br>
 <a href="https://arxiv.org/abs/1608.06993" >https://arxiv.org/abs/1608.06993</a> <br>

 9. Tsung-Yi. et al, Focal Loss for Dense Object Detection <br>
 <a href="https://arxiv.org/abs/1708.02002" >https://arxiv.org/abs/1708.02002</a> <br>
 
 10. O Russakovsky et al, ImageNet Large Scale Visual Recognition Challenge <br>
 <a href="https://arxiv.org/abs/1409.0575" >https://arxiv.org/abs/1409.0575</a> <br>
 
 11. TY Lin et al, Microsoft COCO: Common Objects in Context <br>
 <a href="https://arxiv.org/abs/1405.0312" >https://arxiv.org/abs/1405.0312</a> <br>
 
 12. Moses & John Olafenwa, A collection of images of identifiable professionals.<br>
 <a href="https://github.com/OlafenwaMoses/IdenProf" >https://github.com/OlafenwaMoses/IdenProf</a> <br>
 
 13. Joseph Redmon and Ali Farhadi, YOLOv3: An Incremental Improvement.<br>
 <a href="https://arxiv.org/abs/1804.02767">https://arxiv.org/abs/1804.02767</a> <br>
 
 14. Experiencor, Training and Detecting Objects with YOLO3<br>
 <a href="https://github.com/experiencor/keras-yolo3">https://github.com/experiencor/keras-yolo3</a> <br><br>
 
 
 
 
 
