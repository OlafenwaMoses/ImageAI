from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()

prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "idenprof_resnet_ex-056_acc-0.993062.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3
prediction.setJsonPath(os.path.join(execution_path, "idenprof.json"))
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "9.jpg"), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)