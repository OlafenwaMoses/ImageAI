from imageai.Prediction.Custom import ModelTraining
import os

trainer = ModelTraining()
trainer.setModelTypeAsDenseNet()
trainer.setDataDirectory("idenprof")
trainer.trainModel(num_objects=10, num_experiments=50, enhance_data=True, batch_size=8, show_network_summary=True, continue_from_model="idenprof_densenet-0.763500.h5") # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/models-v3

