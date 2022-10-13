from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

from .utils import transform_prediction


def noop(x):
    return x

class DetectionLayer(nn.Module):

    def __init__(
            self,
            anchors : Union[List[int], Tuple[int, ...]],
            anchor_masks : Tuple[int, int, int],
            layer : int,
            num_classes : int=80,
            device : str="cpu"
        ):
        super().__init__()
        self.height = 416
        self.width = 416
        self.num_classes = num_classes
        self.ignore_thresh = 0.7
        self.truth_thresh = 1
        self.rescore = 1
        self.device = device
        self.anchors = self.__get_anchors(anchors, anchor_masks)
        self.layer = layer
        self.layer_width = None
        self.layer_height = None
        self.layer_output = None
        self.pred = None
        self.stride = None
        self.grid = None
        self.anchor_grid = None

    def __get_anchors(
                self, anchors : Union[List[int], Tuple[int, ...]],
                anchor_masks : Tuple[int, int, int]
            ) -> torch.Tensor:
        a = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
        return torch.tensor([a[i] for i in anchor_masks]).to(self.device)

    def forward(self, x : torch.Tensor):
        self.layer_height, self.layer_width = x.shape[2], x.shape[3]
        self.stride = self.height // self.layer_height
        if self.training:
            batch_size = x.shape[0]
            grid_size = x.shape[2]
            bbox_attrs = 5 + self.num_classes
            num_anchors = len(self.anchors)

            # transform input shape
            self.layer_output = x.detach()
            self.pred = x.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
            
            self.layer_output = self.layer_output.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
            self.layer_output = self.layer_output.transpose(1, 2).contiguous()
            self.layer_output = self.layer_output.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

        else:
            # transform the output of the network and scale it to match the
            # network dimension : 416x416
            self.layer_output =  transform_prediction(
                        x.data, self.width, self.anchors, self.num_classes,
                        self.device
                    )
        return self.layer_output


class ConvLayer(nn.Module):

    def __init__(self, in_f : int, out_f : int, kernel_size : int = 3,
                stride : int = 1, use_batch_norm : bool = True,
                activation : str ="leaky"):
        super().__init__()
        self.conv = nn.Conv2d(
                in_f, out_f, stride=stride, kernel_size=kernel_size,
                padding= kernel_size//2,
                bias=False if use_batch_norm else True
            )
        self.batch_norm = nn.BatchNorm2d(out_f) if use_batch_norm else noop
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True) if activation=="leaky" else noop

    def forward(self, x : torch.Tensor):
        return self.leaky_relu(self.batch_norm(self.conv(x)))

class YoloV3(nn.Module):

    def __init__(
            self,
            anchors : Union[List[int], Tuple[int, ...]],
            num_classes : int = 80,
            device : str ="cpu"):
        super().__init__()

        # Network Layers
        self.conv1 = ConvLayer(3, 32)
        self.conv2 = ConvLayer(32, 64, stride=2)
        self.conv3 = ConvLayer(64, 32, 1, 1)
        self.conv4 = ConvLayer(32, 64)
        # self.__shortcut_layer1(self.conv4, self.conv2)
        self.conv5 = ConvLayer(64, 128, stride=2)
        self.conv6 = ConvLayer(128, 64, 1, 1)
        self.conv7 = ConvLayer(64, 128, stride=1)
        # self.__shortcut_layer2(self.conv7, self.conv5)
        self.conv8 = ConvLayer(128, 64, 1, 1)
        self.conv9 = ConvLayer(64, 128, stride=1)
        # self.__shortcut_layer3(self.conv9, shortcut2)
        self.conv10 = ConvLayer(128, 256, stride=2)
        self.conv11 = ConvLayer(256, 128, 1, 1)
        self.conv12 = ConvLayer(128, 256)
        # self.__shortcut_layer4(self.con12, self.conv10)
        self.conv13 = ConvLayer(256, 128, 1, 1)
        self.conv14 = ConvLayer(128, 256)
        # self.__shortcut_layer5(self.conv14, shortcut4)
        self.conv15 = ConvLayer(256, 128, 1, 1)
        self.conv16 = ConvLayer(128, 256)
        # self.__shortcut_layer6(self.conv16, shortcut5)
        self.conv17 = ConvLayer(256, 128, 1, 1)
        self.conv18 = ConvLayer(128, 256)
        # self.__shortcut_layer7(self.conv18, shortcut6)
        self.conv19 = ConvLayer(256, 128, 1, 1)
        self.conv20 = ConvLayer(128, 256)
        # self.__shortcut_layer8(self.conv20, shortcut7)
        self.conv21 = ConvLayer(256, 128, 1, 1)
        self.conv22 = ConvLayer(128, 256)
        # self.__shortcut_layer9(self.conv22, shortcut8)
        self.conv23 = ConvLayer(256, 128, 1, 1)
        self.conv24 = ConvLayer(128, 256)
        # self.__shortcut_layer10(self.conv24, shortcut9)
        self.conv25 = ConvLayer(256, 128, 1, 1)
        self.conv26 = ConvLayer(128, 256)
        # self.__shortcut_layer11(self.conv26, shortcut10)
        self.conv27 = ConvLayer(256, 512, stride=2)
        self.conv28 = ConvLayer(512, 256, 1, 1)
        self.conv29 = ConvLayer(256, 512)
        # self.__shortcut_layer12(self.conv29, self.conv27)
        self.conv30 = ConvLayer(512, 256, 1, 1)
        self.conv31 = ConvLayer(256, 512)
        # self.__shortcut_layer13(self.conv31, shortcut12)
        self.conv32 = ConvLayer(512, 256, 1, 1)
        self.conv33 = ConvLayer(256, 512)
        # self.__shortcut_layer14(self.conv33, shortcut13)
        self.conv34 = ConvLayer(512, 256, 1, 1)
        self.conv35 = ConvLayer(256, 512)
        # self.__shortcut_layer15(self.conv35, shortcut14)
        self.conv36 = ConvLayer(512, 256, 1, 1)
        self.conv37 = ConvLayer(256, 512)
        # self.__shortcut_layer16(self.conv37, shortcut15)
        self.conv38 = ConvLayer(512, 256, 1, 1)
        self.conv39 = ConvLayer(256, 512)
        # self.__shortcut_layer17(self.conv39, shortcut16)
        self.conv40 = ConvLayer(512, 256, 1, 1)
        self.conv41 = ConvLayer(256, 512)
        # self.__shortcut_layer18(self.conv41, shortcut17)
        self.conv42 = ConvLayer(512, 256, 1, 1)
        self.conv43 = ConvLayer(256, 512)
        # self.__shortcut_layer19(self.conv43, shortcut18)
        self.conv44 = ConvLayer(512, 1024, stride=2)
        self.conv45 = ConvLayer(1024, 512, 1, 1)
        self.conv46 = ConvLayer(512, 1024)
        # self.__shortcut_layer20(self.conv46, self.conv44)
        self.conv47 = ConvLayer(1024, 512, 1, 1)
        self.conv48 = ConvLayer(512, 1024)
        # self.__shortcut_layer21(self.conv48, shortcut20)
        self.conv49 = ConvLayer(1024, 512, 1, 1)
        self.conv50 = ConvLayer(512, 1024)
        # self.__shortcut_layer22(self.conv50, shortcut21)
        self.conv51 = ConvLayer(1024, 512, 1, 1)
        self.conv52 = ConvLayer(512, 1024)
        # self.__shortcut_layer23(self.conv52, shortcut22)
        self.conv53 = ConvLayer(1024, 512, 1, 1)
        self.conv54 = ConvLayer(512, 1024)
        self.conv55 = ConvLayer(1024, 512, 1, 1)
        self.conv56 = ConvLayer(512, 1024)
        self.conv57 = ConvLayer(1024, 512, 1, 1)
        self.conv58 = ConvLayer(512, 1024)
        self.conv59 = ConvLayer(
                    1024, (3 * (5 + num_classes)), 1, 1, use_batch_norm=False,
                    activation="linear"
                )

        # yolo layer
        self.yolo1 = DetectionLayer(
                    num_classes=num_classes, anchors=anchors,
                    anchor_masks=(6, 7, 8), device=device, layer=1
                )

        # self.__route_layer(self.conv57)
        self.conv60 = ConvLayer(512, 256, 1, 1)
        self.upsample1 = nn.Upsample(
                    scale_factor=2, mode="nearest"
                    #align_corners=True
                )
        # self.__route_layer(self.upsample1, shortcut19)
        self.conv61 = ConvLayer(768, 256, 1, 1)
        self.conv62 = ConvLayer(256, 512)
        self.conv63 = ConvLayer(512, 256, 1, 1)
        self.conv64 = ConvLayer(256, 512)
        self.conv65 = ConvLayer(512, 256, 1, 1)
        self.conv66 = ConvLayer(256, 512)
        self.conv67 = ConvLayer(
                    512, (3 * (5 + num_classes)), 1, 1, use_batch_norm=False,
                    activation="linear"
                )
        
        # yolo layer
        self.yolo2 = DetectionLayer(
                    num_classes=num_classes, anchors=anchors,
                    anchor_masks=(3, 4, 5), device=device, layer=2
                )
        
        # self.__route_layer(self.conv65)
        self.conv68 = ConvLayer(256, 128, 1, 1)
        self.upsample2 = nn.Upsample(
                    scale_factor=2, mode="nearest"
                    #align_corners=True
                )
        # self.__route_layer(self.upsample2, shortcut11)

        self.conv69 = ConvLayer(384, 128, 1, 1)
        self.conv70 = ConvLayer(128, 256)
        self.conv71 = ConvLayer(256, 128, 1, 1)
        self.conv72 = ConvLayer(128, 256)
        self.conv73 = ConvLayer(256, 128, 1, 1)
        self.conv74 = ConvLayer(128, 256)
        self.conv75 = ConvLayer(
                    256, (3 * (5 + num_classes)), 1, 1, use_batch_norm=False,
                    activation="linear"
                )

        # yolo layer
        self.yolo3 = DetectionLayer(
                    num_classes=num_classes, anchors=anchors,
                    anchor_masks=(0, 1, 2), device=device, layer=3
                )
    
    def get_loss_layers(self) -> List[torch.Tensor]:
        return [self.yolo1, self.yolo2, self.yolo3]

    def __route_layer(self, y1 : torch.Tensor, y2 : Optional[torch.Tensor]=None):
        if isinstance(y2, torch.Tensor):
            return torch.cat([y1, y2], 1)
        return y1

    def __shortcut_layer(self,
                         y1 : torch.Tensor, y2 : torch.Tensor,
                         activation : str="linear"
                        ) -> torch.Tensor:
        actv = noop if activation=="linear" else nn.LeakyReLU(0.1)
        return actv(y1 + y2)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        # shortcut1
        y = self.conv5(self.__shortcut_layer(self.conv4(self.conv3(y)), y))
        y2 = self.conv7(self.conv6(y))
        # shortcut2
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv9(self.conv8(y))
        # shortcut3
        y2 = self.conv10(self.__shortcut_layer(y2, y))
        y = self.conv12(self.conv11(y2))
        # shortcut4
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv14(self.conv13(y2))
        # shortcut5
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv16(self.conv15(self.__shortcut_layer(y2, y)))
        # shortcut6
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv18(self.conv17(y2))
        # shortcut7
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv20(self.conv19(y2))
        # shortcut8
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv22(self.conv21(y2))
        # shortcut9
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv24(self.conv23(y2))
        # shortcut10
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv26(self.conv25(y2))
        # shortcut11
        r1 = self.__shortcut_layer(y, y2) # route_layer
        y = self.conv27(r1)
        y2 = self.conv29(self.conv28(y))
        # shortcut12
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv31(self.conv30(y))
        # shortcut13
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv33(self.conv32(y))
        # shortcut14
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv35(self.conv34(y))
        # shortcut15
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv37(self.conv36(y))
        # shortcut16
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv39(self.conv38(y))
        # shortcut17
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv41(self.conv40(y))
        # shortcut18
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv43(self.conv42(y))
        # shortcut19
        r2 = self.__shortcut_layer(y2, y) # route_layer
        y2 = self.conv44(r2)
        y = self.conv46(self.conv45(y2))
        # shortcut20
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv48(self.conv47(y2))
        # shortcut21
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv50(self.conv49(y2))
        # shortcut22
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv52(self.conv51(y2))
        # shortcut23
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv54(self.conv53(y2))
        r3 = self.conv57(self.conv56(self.conv55(y))) # route_layer
        y = self.conv59(self.conv58(r3))

        # first detection layer
        out = self.yolo1(y)
        y = self.conv60(self.__route_layer(r3))
        y = self.conv62(self.conv61(self.__route_layer(self.upsample1(y), r2)))
        r4 = self.conv65(self.conv64(self.conv63(y))) # route_layer
        y = self.conv67(self.conv66(r4))

        # second detection layer
        out = torch.cat([out, self.yolo2(y)], dim=1)
        y = self.conv68(self.__route_layer(r4))
        y = self.conv70(self.conv69(self.__route_layer(self.upsample2(y), r1)))
        y = self.conv75(self.conv74(self.conv73(self.conv72(self.conv71(y)))))

        # third detection layer
        out = torch.cat([out, self.yolo3(y)], dim=1)

        return out
