import os

from imageai.Prediction import ImagePrediction

PATH_HERE = os.getcwd()
PATH_MODEL = os.path.join(PATH_HERE, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")


def main(path_model, path_dir):
    multiple_prediction = ImagePrediction()
    multiple_prediction.setModelTypeAsResNet()
    multiple_prediction.setModelPath(path_model)
    multiple_prediction.loadModel()

    all_images_array = []

    all_files = os.listdir(path_dir)
    for each_file in all_files:
        if each_file.endswith(".jpg" or each_file.endswith(".png")):
            all_images_array.append(each_file)

    results_array = multiple_prediction.predictMultipleImages(all_images_array, result_count_per_image=5)

    for each_result in results_array:
        for idx in range(len(each_result["predictions"])):
            print(each_result["predictions"][idx], " : ", each_result["percentage_probabilities"][idx])
        print("-----------------------")


if __name__ == '__main__':
    main(path_model=PATH_MODEL, path_dir=PATH_HERE)