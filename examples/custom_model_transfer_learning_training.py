from imageai.Prediction.Custom import ModelTraining
import os

trainer = ModelTraining()
trainer.setModelTypeAsResNet()
trainer.setDataDirectory("idenprof")
trainer.trainModel(num_objects=10, num_experiments=50, enhance_data=True, batch_size=16, show_network_summary=True,transfer_from_model="resnet50_weights_tf_dim_ordering_tf_kernels.h5", initial_num_objects=1000) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3
