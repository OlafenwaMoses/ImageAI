import os

def extension_check(file_path: str):
    if file_path.endswith(".h5"):
        raise RuntimeError("You are trying to use a Tensorflow model with ImageAI. ImageAI now uses PyTorch as backed as from version 3.0.2 . If you want to use the Tensorflow models or a customly trained '.h5' model, install ImageAI 2.1.6 or earlier. To use the latest Pytorch models, see the documentation in https://imageai.readthedocs.io/")
    elif file_path.endswith(".pt") == False and file_path.endswith(".pth") == False:
        raise ValueError(f"Invalid model file {os.path.basename(file_path)}. Please parse in a '.pt' and '.pth' model file.")
