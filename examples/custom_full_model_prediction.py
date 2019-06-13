from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()


predictor = CustomImagePrediction()
predictor.setModelPath(model_path=os.path.join(execution_path, "idenprof_full_resnet_ex-001_acc-0.119792.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3
predictor.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor.loadFullModel(num_objects=10)


results, probabilities = predictor.predictImage(image_input=os.path.join(execution_path, "1.jpg"), result_count=5)
print(results)
print(probabilities)



