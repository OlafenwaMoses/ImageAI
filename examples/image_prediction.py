import os

from imageai.Prediction import ImagePrediction

PATH_HERE = os.getcwd()
PATH_MODEL = os.path.join(PATH_HERE, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")
PATH_IMAGE_INPUT = os.path.join(PATH_HERE, "image1.jpg")


def main(path_model, path_img):
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(path_model)
    prediction.loadModel()

    predictions, probabilities = prediction.predictImage(path_img, result_count=10)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction , " : " , eachProbability)


if __name__ == '__main__':
    main(path_model=PATH_MODEL, path_img=PATH_IMAGE_INPUT)