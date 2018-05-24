from imageai.Prediction.Custom import ModelTraining


model_trainer = ModelTraining()
model_trainer.setModelTypeAsSqueezeNet()
model_trainer.setDataDirectory("C:\\Users\\Moses\\Documents\\Moses\\W7\\AI\\Custom Datasets\\IDENPROF\\idenprof-small-test\\idenprof")
model_trainer.trainModel(num_objects=10, num_experiments=20, enhance_data=True, batch_size=4, show_network_summary=True)
