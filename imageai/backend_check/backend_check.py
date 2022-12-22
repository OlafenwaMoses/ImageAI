try:
    import torch
    import torchvision
except:
    try:
        import tensorflow
        import keras

        raise RuntimeError("Dependency error!!! It appears you are trying to use ImageAI with a Tensorflow backend. ImageAI now uses PyTorch as backed as from version 3.0.2 . If you want to use the Tensorflow models or a customly trained '.h5' model, install ImageAI 2.1.6 or earlier. To use the latest Pytorch models, see the documentation in https://imageai.readthedocs.io/")
    except:
        raise RuntimeError("Dependency error!!! PyTorch and TorchVision are not installed. Please see installation instructions in the documentation https://imageai.readthedocs.io/")