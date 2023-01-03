# Overview

In December 2022, ImageAI `3.0.2` was released which effected the change from Tensorflow backend to PyTorch backend. This change allows ImageAI to support `Python 3.7` up to `Python 3.10` for all its features and deprecates a number of functionalities for this and future versions of ImageAI.


# Deprecated functionalities
- Tensorflow backend no longer supported. Now replaced with PyTorch
- All `.h5` pretrained models and custom trained `.h5` models no longer supported. If you still intend to use these models, see the `Using Tensorflow backend` section.
- `Speed mode` have been removed from model loading
- Custom detection model training dataset format changed to YOLO format from Pascal VOC. To convert your dataset to YOLO format, see the  `Convert Pascal VOC dataset to YOLO format` section.
- Enhance data for custom classification model training now removed
- Detection model training standalone evaluation now removed

# Using Tensorflow backend
To use Tensorflow backend, do the following

- Install Python 3.7
- Install Tensorflow 
  - CPU: `pip install tensorflow==2.4.0`
  - GPU: `pip install tensorflow-gpu==2.4.0`
- Install other dependencies: `pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0`
- Install ImageAI **2.1.6**: `pip install imageai==2.1.6`
- Download the Tensorflow models from the releases below
  - [Models for Image Recognition and Object Detection](https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0)
  - [TF2.x Models [ Exclusives ]](https://github.com/OlafenwaMoses/ImageAI/releases/tag/essentials-v5)



# Convert Pascal VOC dataset to YOLO format
Because ImageAI now uses `YOLO format` for training custom object detection models; should you need to train a new model with the new ImageAI version, you will need to convert your `Pascal VOC` datasets to YOLO format by doing the following 
- Run the command below
    ```
    python scripts/pascal_voc_to_yolo.py --dataset_dir <path_to_your_dataset_folder>
    ```
- Once completed, you will find the YOLO version of the dataset next to your Pascal VOC dataset.
  - E.g, if your dataset is in `C:/Users/Troublemaker/Documents/datasets/headset`, your conversion command will be
    ```
    python scripts/pascal_voc_to_yolo.py --dataset_dir C:/Users/Troublemaker/Documents/datasets/headset
    ```
    and once completed, the output will be in `C:/Users/Troublemaker/Documents/datasets/headset-yolo`
