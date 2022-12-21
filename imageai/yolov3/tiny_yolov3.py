from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

from .yolov3 import DetectionLayer, ConvLayer


class YoloV3Tiny(nn.Module):

    def __init__(
                self,
                anchors : Union[List[int], Tuple[int,...]],
                num_classes : int=80,
                device : str="cpu"
            ):
        super().__init__()

        # Network Layers
        self.conv1 = ConvLayer(3, 16)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvLayer(16, 32)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvLayer(32, 64)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvLayer(64, 128)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvLayer(128, 256)
        self.maxpool5 = nn.MaxPool2d(2, 2)
        self.conv6 = ConvLayer(256, 512)
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1))
        self.maxpool6 = nn.MaxPool2d(2, 1)
        self.conv7 = ConvLayer(512, 1024)
        self.conv8 = ConvLayer(1024, 256, 1, 1)
        self.conv9 = ConvLayer(256, 512)
        self.conv10 = ConvLayer(
                    512, (3 * (5+num_classes)), 1, 1,
                    use_batch_norm=False,
                    activation="linear"
                )
        self.yolo1 = DetectionLayer(
                    num_classes=num_classes, anchors=anchors,
                    anchor_masks=(3, 4, 5), device=device, layer=1
                )
        # self.__route_layer(conv8)
        self.conv11 = ConvLayer(256, 128, 1, 1)
        self.upsample1 = nn.Upsample(
                    scale_factor=2, mode="nearest"
                    #align_corners=True
                )
        # self.__route_layer(upsample1, conv5)
        self.conv12 = ConvLayer(384, 256)
        self.conv13 = ConvLayer(
                    256, (3 * (5 + num_classes)), 1, 1,
                    use_batch_norm=False,
                    activation="linear"
                )
        self.yolo2 = DetectionLayer(
                    num_classes=num_classes, anchors=anchors,
                    anchor_masks=(0, 1, 2), device=device, layer=2
                )
    
    def get_loss_layers(self) -> List[torch.Tensor]:
        return [self.yolo1, self.yolo2]

    def __route_layer(self, y1 : torch.Tensor, y2 : Optional[torch.Tensor]=None) -> torch.Tensor:
        if isinstance(y2, torch.Tensor):
            return torch.cat([y1, y2], 1)
        return y1

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        y = self.maxpool2(self.conv2(self.maxpool1(self.conv1(x))))
        y = self.maxpool4(self.conv4(self.maxpool3(self.conv3(y))))
        r1 = self.conv5(y) # route layer
        y = self.zeropad(self.conv6(self.maxpool5(r1)))
        y = self.conv7(self.maxpool6(y))
        r2 = self.conv8(y) # route layer
        y = self.conv10(self.conv9(r2))

        # first detection layer
        out = self.yolo1(y)
        y = self.conv11(self.__route_layer(r2))
        y = self.__route_layer(self.upsample1(y), r1)
        y = self.conv13(self.conv12(y))
        
        # second detection layer
        out = torch.cat([out, self.yolo2(y)], 1)

        return out
