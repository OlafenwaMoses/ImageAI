import os

import numpy as np
import torch
from torchvision.ops import box_iou

from ....yolov3.utils import get_predictions
from .metric import ap_per_class
from tqdm import tqdm

# This new validation function is based on https://github.com/ultralytics/yolov3/blob/master/val.py


def xywh2xyxy(box_coord : torch.Tensor):
    """
    Convert bounding box coordinates from center_x, center_y, width, height
    to x_1, y_1, x_2, x_3
    """
    n = box_coord.clone()
    n[:, 0] = (box_coord[:, 0] - (box_coord[:, 2] / 2))
    n[:, 1] = (box_coord[:, 1] - (box_coord[:, 3] / 2))
    n[:, 2] = (box_coord[:, 0] + (box_coord[:, 2] / 2))
    n[:, 3] = (box_coord[:, 1] + (box_coord[:, 3] / 2))

    return n

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    detections[:, [1,3]] = torch.clamp(detections[:, [1,3]], 0.0, 416)
    detections[:, [2,4]] = torch.clamp(detections[:, [2,4]], 0.0, 416)
    
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, 1:5])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 7]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

@torch.no_grad()
def run(model, val_dataloader, num_class, net_dim=416, nms_thresh=0.6, objectness_thresh=0.001, device="cpu"):
    model.eval()
    nc = int(num_class)  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap, ap_class = [], [], []
 
    for batch_i, (im, targets) in tqdm(enumerate(val_dataloader)):
        im = im.to(device)
        targets = targets.to(device)
        nb = im.shape[0]  # batch

        # Inference
        out = model(im) # inference

        # NMS
        targets[:, 2:] *= torch.Tensor([net_dim, net_dim, net_dim, net_dim]).to(device)  # to pixels
        out = get_predictions(
                pred=out.to(device), num_classes=nc,
                objectness_confidence=objectness_thresh,
                nms_confidence_level=nms_thresh, device=device
            )

        # Metrics
        for si in range(nb):
            labels = targets[targets[:, 0] == si, 1:]
            pred = out[out[:, 0]==si, :] if isinstance(out, torch.Tensor) else torch.zeros((0,0), device=device)
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool, device="cpu"), torch.Tensor(device="cpu"), torch.Tensor(device="cpu"), tcls))
                continue

            # Predictions
            if nc==1:
                pred[:, 7] = 0
            
            if pred.shape[0] > 300:
                pred = pred[:300, :]  # sorted by confidence
                
            predn = pred.clone()

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5]).to(device)  # target boxes
                labelsn = torch.cat((labels[:, 0:1], tbox), 1).to(device)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 7].cpu(), tcls))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    return mp, mr, map50, map

