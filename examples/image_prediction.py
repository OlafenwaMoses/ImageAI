from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "1.jpg"), result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)