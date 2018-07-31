# ImageAI : Custom Image Prediction <br>
<p>An <b>AI Commons</b> project <a href="https://commons.specpal.science" >https://commons.specpal.science </a></p>
<hr>
<br>
<br>
      ImageAI provides 4 different algorithms and model types to perform custom image prediction using your custom models.
You will be able to use your model trained with <b>ImageAI</b> and the corresponding model_class JSON file to predict custom objects
that you have trained the model on. In this example, we will be using the model trained for 20 experiments on <b>IdenProf</b>, a dataset
 of uniformed professionals and achieved 65.17% accuracy on the test dataset (You can use your own trained model and generated JSON file. This 'class' is provided mainly for the purpose to use your own custom models.).  Download the ResNet model of the model and JSON files in links below: <br>
       <span><b>- <a href="https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof_061-0.7933.h5" style="text-decoration: none;" >ResNet</a></b> (Size = 90.4 mb)</span> <br>
        <span><b>- <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0.1/model_class.json" style="text-decoration: none;" >IdenProf model_class.json file</a></b> </span> <br>
       Great! Once you have downloaded this model file and the JSON file, start a new python project, and then copy the model file
and the JSON file to your project folder where your python files (.py files) will be . Download the image below, or take any image on your computer
 that include any of the following professionals(Chef, Doctor, Engineer, Farmer, Fireman, Judge, Mechanic, Pilot, Police and Waiter)
and copy it to your python project's folder. Then create a python file and give it a name; an example is <b>FirstCustomPrediction.py</b>.
      Then write the code below into the python file: <br><br>
<h3><b>FirstCustomPrediction.py</b></h3>
<b><pre>from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "idenprof_061-0.7933.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "4.jpg"), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)</pre></b>


<p>Sample Result:
    <br>
    <img src="../../images/4.jpg" style="width: 400px; height: auto;" />
    <pre>mechanic : 76.82620286941528
chef : 10.106072574853897
waiter : 4.036874696612358
police : 2.6663416996598244
pilot : 2.239348366856575</pre>
</p>

<br>

The code above works as follows: <br>
     <b><pre>from imageai.Prediction.Custom import CustomImagePrediction
import os</pre></b>
<br>
      The code above imports the <b>ImageAI</b> library for custom image prediction
 and the python <b>os</b> class. <br>
<b><pre>execution_path = os.getcwd()</pre></b>
<br> The above line obtains the path to the folder that contains
your python file (in this example, your FirstCustomPrediction.py) . <br>

<b><pre>prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet_model_ex-020_acc-0.651714.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=10)
</pre></b>
      In the lines above, we created and instance of the <b>CustomImagePrediction()</b>
 class in the first line, then we set the model type of the prediction object to ResNet by caling the <b>.setModelTypeAsResNet()</b>
  in the second line, we set the model path of the prediction object to the path of the custom model file (<b>resnet_model_ex-020_acc-0.651714.h5</b>) we copied to the python file folder
   in the third line, we set the path to  the model_class.json of the model, we load the model and parse the number of objected that can be predicted in the model.

<b><pre>predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "4.jpg"), result_count=5)</pre></b> In the above line, we defined 2 variables to be equal to the function
 called to predict an image, which is the <b>.predictImage()</b> function, into which we parsed the path to
 our image and also state the number of prediction results we want to have (values from 1 to 10 in this case) parsing
 <b> result_count=5 </b>. The <b>.predictImage()</b> function will return 2 array objects with the first (<b>predictions</b>) being
  an array of predictions and the second (<b>percentage_probabilities</b>) being an array of the corresponding percentage probability for each
  prediction.

  <b><pre>for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)</pre></b> The above line obtains each object in the <b>predictions</b> array, and also
obtains the corresponding percentage probability from the <b>percentage_probabilities</b>, and finally prints
the result of both to console.

</span>


<br><br>

<b>CustomImagePrediction</b> class also supports the multiple predictions, input types and prediction speeds that are contained
in the <b>ImagePrediction</b> class. Follow this <a href="README.md" >link</a> to see all the details.


<h3><b><u> >> Documentation</u></b></h3>
We have provided full documentation for all <b>ImageAI</b> classes and functions in 2 major languages. Find links below: <br>

<b> >> Documentation - English Version  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)</b> <br>
<b> >> Documentation - Chinese Version  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)</b>

