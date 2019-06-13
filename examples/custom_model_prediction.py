from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "idenprof_resnet.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3
prediction.setJsonPath(os.path.join(execution_path, "idenprof.json"))
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "9.jpg"), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)