from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()


predictor = CustomImagePrediction()
predictor.setModelTypeAsResNet()
predictor.setModelPath(model_path=os.path.join(execution_path, "idenprof_resnet.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3
predictor.setJsonPath(model_json=os.path.join(execution_path, "idenprof.json"))
predictor.loadModel(num_objects=10)
predictor.save_model_to_tensorflow(new_model_folder= os.path.join(execution_path, "tensorflow_model"), new_model_name="idenprof_resnet_tensorflow.pb")




