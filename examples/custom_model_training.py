from imageai.Prediction.Custom import ModelTraining


model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory(r"C:/Users/Moses/Documents/Moses/W7/AI/Custom Datasets/idenprof")
model_trainer.trainModel(num_objects=10, num_experiments=20, enhance_data=True, batch_size=32, show_network_summary=True)
