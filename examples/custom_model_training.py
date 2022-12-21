from imageai.Classification.Custom import ClassificationModelTrainer


model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory("idenprof")
model_trainer.trainModel(num_experiments=200, batch_size=32)
