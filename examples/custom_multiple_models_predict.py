from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()


predictor = CustomImagePrediction()
predictor.setModelPath(model_path=os.path.join(execution_path, "idenprof_resnet.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3
predictor.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor.setModelTypeAsResNet()
predictor.loadModel(num_objects=10)


predictor2 = CustomImagePrediction()
predictor2.setModelPath(model_path=os.path.join(execution_path, "idenprof_full_resnet_ex-001_acc-0.119792.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3
predictor2.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor2.loadFullModel(num_objects=10)

predictor3 = CustomImagePrediction()
predictor3.setModelPath(model_path=os.path.join(execution_path, "idenprof_inception_0.719500.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3
predictor3.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor3.setModelTypeAsInceptionV3()
predictor3.loadModel(num_objects=10)

results, probabilities = predictor.predictImage(image_input=os.path.join(execution_path, "9.jpg"), result_count=5)
print(results)
print(probabilities)

results2, probabilities2 = predictor2.predictImage(image_input=os.path.join(execution_path, "9.jpg"),
                                                    result_count=5)
print(results2)
print(probabilities2)

results3, probabilities3 = predictor3.predictImage(image_input=os.path.join(execution_path, "9.jpg"),
                                                       result_count=5)
print(results3)
print(probabilities3)
print("-------------------------------")


