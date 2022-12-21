from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/resnet50-19c8e357.pth
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "1.jpg"), result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)