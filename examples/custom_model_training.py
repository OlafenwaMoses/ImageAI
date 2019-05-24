from imageai.Prediction.Custom import ModelTraining

PATH_DATA = r"C:/Users/Moses/Documents/Moses/W7/AI/Custom Datasets/idenprof"


def main(path_data):
    model_trainer = ModelTraining()
    model_trainer.setModelTypeAsResNet()
    model_trainer.setDataDirectory(path_data)
    model_trainer.trainModel(num_objects=10, num_experiments=20, enhance_data=True,
                             batch_size=32, show_network_summary=True)


if __name__ == '__main__':
    main(path_data=PATH_DATA)