from imageai.Prediction.Custom import CustomImagePrediction
import os

PATH_HERE = os.getcwd()
PATH_MODEL = os.path.join(PATH_HERE, "resnet_model_ex-020_acc-0.651714.h5")
PATH_CLASSES = os.path.join(PATH_HERE, "model_class.json")
PATH_IMAGE = os.path.join(PATH_HERE, "4.jpg")


def main(path_model, path_classes, path_image):
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(path_model)
    prediction.setJsonPath(path_classes)
    prediction.loadModel(num_objects=10)

    predictions, probabilities = prediction.predictImage(path_image, result_count=5)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction , " : " , eachProbability)


if __name__ == '__main__':
    main(path_model=PATH_MODEL, path_classes=PATH_CLASSES, path_image=PATH_IMAGE)
