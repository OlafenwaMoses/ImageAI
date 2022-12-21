# Overview

In December 2022, ImageAI `3.0.0` was released which effected the change from Tensorflow backend to PyTorch backend. This change allows ImageAI to support `Python 3.7` up to `Python 3.10` for all its features and deprecates a number of functionalities for this and future versions of ImageAI.


# Deprecated functionalities
- Tensorflow backend no longer supported. Now replaced with PyTorch
- All `.h5` pretrained models and custom trained `.h5` models no longer supported. If you still intend to use these models, see the `Using Tensorflow backend` section.
- `Speed mode` have been removed from model loading
- Custom detection model training dataset format changed to YOLO format from Pascal VOC. To convert your dataset to YOLO format, see the  `Convert Pascal VOC dataset to YOLO format` section.

# Using Tensorflow backend
To use Tensorflow backend, do the following

- Install Python 3.7
- Install Tensorflow 
  - CPU: `pip install tensorflow==2.4.0`
  - GPU: `pip install tensorflow-gpu==2.4.0`
- Install other dependencies: `pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0`
- Install ImageAI **2.1.6**: `pip install imageai==2.1.6`



# Convert Pascal VOC dataset to YOLO format
