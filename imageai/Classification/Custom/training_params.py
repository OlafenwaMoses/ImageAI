import torch
from torch.optim import SGD
from torchvision.models import resnet50

model = resnet50(pretrained=False)


def resnet50_train_params():
    model = resnet50(pretrained=False)
    return {
            "model": model,
            "optimizer": SGD,
            "weight_decay":1e-4,
            "lr":0.1,
            "lr_decay_rate": None,
            "lr_step_size":None
        }
