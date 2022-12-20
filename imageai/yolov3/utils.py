import math
from typing import Union, List, Tuple

import torch
import numpy as np
import cv2 as cv

def draw_bbox_and_label(x : torch.Tensor, label : str, img : np.ndarray) -> np.ndarray:
    """
    Draws the predicted bounding boxes on the original image.
    """
    x1,y1,x2,y2 = tuple(map(int, x))
    if x is not None:
        img = cv.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = (x1 + t_size[0] + 3, y1 + t_size[1] + 4)
    img = cv.putText(img, label, (x1, y1+t_size[1]+4), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)

    return img 

def letterbox_image(
        image : np.ndarray,
        inp_dim : Tuple[int, int]) -> np.ndarray:
    """
    Resizes images into the dimension expected by the network. This
    function fills extra spaces in the image with grayscale, if the
    image is smaller than the expected dimesion. This implementation
    keeps the aspect ration of the original image.
    """
    img_w, img_h = image.shape[1], image.shape[0] # original image dimension
    net_w, net_h = inp_dim # the dimension expected by the network.

    # calculate the new dimension with same aspect ration as
    # the original image.
    scale_factor = min(net_w/img_w, net_h/img_h)
    new_w = int(round(img_w * scale_factor))
    new_h = int(round(img_h * scale_factor))

    resized_image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    canvas = np.full((net_w, net_h, 3), 128)
    canvas[(net_h - new_h)//2 : (net_h - new_h)//2 + new_h, (net_w - new_w)//2 : (net_w - new_w)//2 + new_w, :] = resized_image
    return canvas

def prepare_image(
        image : np.ndarray,
        inp_dim : Tuple[int, int]) -> torch.Tensor:
    """
    Prepared the input to match the expectation of the network.
    """
    img = letterbox_image(image, inp_dim)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def bbox_iou(bbox1 : torch.Tensor, bbox2 : torch.Tensor, device="cpu"):
    """
    Returns the IoU value of overlapping boxes
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    # intersections
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.max(inter_rect_x2 - inter_rect_x1+1, torch.zeros(inter_rect_x2.shape, device=device)) * \
                torch.max(inter_rect_y2 - inter_rect_y1+1, torch.zeros(inter_rect_y2.shape, device=device))

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    return inter_area / (b1_area + b2_area - inter_area)

def transform_prediction(
        pred : torch.Tensor,
        inp_dim : int,
        anchors : Union[List[int], Tuple[int, ...], torch.Tensor],
        num_classes : int,
        device : str = "cpu"
        ) -> torch.Tensor:
    """
    Transforms the predictions of the convolutional layers
    from
        batch_size x (3 * 5+num_classes) x grid_size x grid_size
    to
        batch_size x (grid_size * grid_size * anchors) x num_classes
    aids the concatenation of the prediction at the three detection layers
    and also for easy representation of the predicted bounding boxes.

    Also, transforms the bounding box predictions and the objectness score
    to match the discription specified in the paper:
        Bx = sigmoid(Tx) + Cx
        By = sigmoid(Ty) + Cy
        Bw = Pw(exp(Tw))
        Bh = Ph(exp(Th))

    Parameters:
    -----------
        pred:           prediction of the convolutional layer
        inp_dim:        the dimension of images expected by the yolo neural network
        anchors:        a list of anchors
        num_classes:    the numbers of unique classes as specified by COCO.

    Returns:
    --------
        the transformed input.
    """
    batch_size = pred.shape[0]
    grid_size = pred.shape[2]
    stride = inp_dim // grid_size
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # transform input shape
    pred = pred.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    pred = pred.transpose(1, 2).contiguous()
    pred = pred.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # since the dimensions of the anchors are in accordance with the original
    # dimension of the image, it's required to scale the dimension of the
    # anchors to match the dimension of the output of the convolutional
    # layer
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # sigmoid the center_x, center_y and the objectness score
    pred[:, :, 0] = torch.sigmoid(pred[:, :, 0])
    pred[:, :, 1] = torch.sigmoid(pred[:, :, 1])
    pred[:, :, 4] = torch.sigmoid(pred[:, :, 4])

    # add the center offsets
    grid = torch.arange(grid_size, dtype=torch.float)
    grid = np.arange(grid_size)
    x_o, y_o = np.meshgrid(grid, grid)
    #x_offset, y_offset = torch.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(x_o).view(-1, 1).to(device)
    y_offset = torch.FloatTensor(y_o).view(-1, 1).to(device)
    #x_offset = x_offset.transpose(0,1).reshape(-1,1).to(device)
    #y_offset = y_offset.transpose(0,1).reshape(-1,1).to(device)

    x_y_offset = torch.cat([x_offset, y_offset], dim=1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    pred[:, :, :2] += x_y_offset
    
    # transform height and width
    anchors = torch.FloatTensor(anchors).to(device)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    pred[:, :, 2:4] = torch.exp(pred[:, :, 2:4])*anchors

    # apply sigmoid to class scores
    pred[:, :, 5:5+num_classes] = torch.sigmoid(pred[:, :, 5:5+num_classes])

    # resize bounding box prediction to the original image dimension
    pred[:, :, :4] *= stride

    return pred

def get_predictions(
        pred : torch.Tensor,
        num_classes : int,
        objectness_confidence : float = 0.5,
        nms_confidence_level : float = 0.4,
        device : str = "cpu") -> Union[torch.Tensor, int]:
    """
    This function filters the bounding boxes predicted by the network by first
    discarding bounding boxes that has low objectness score, and then proceeds
    to filter overlapping bounding boxes using the non-maximum suppression
    algorithm.

    Parameters:
    -----------
        pred:           a tensor (predicted output) of shape 
                        'batch_size x num_bboxes x bbox_attrs'
        num_classes:    the number of unique classes as provided by COCO.
        objectness_confidence_level:    probability threshold for bounding boxes
                                        containing a valid object.
        nms_convidence_level:           threshold for overlapping bounding boxes

    Returns:
    --------
        The prediction with reasonable bounding boxes.
    """
    conf_mask = (pred[:, :, 4] > objectness_confidence).float().unsqueeze(2)
    pred = pred * conf_mask

    # transform the predicted centers, height and width to top-left corner and
    # right bottom corner coordinates to aid the ease computation of the IoU
    bbox_corner = pred.new(pred.shape)
    bbox_corner[:, :, 0] = (pred[:, :, 0] - (pred[:, :, 2] / 2)) # top-left_x
    bbox_corner[:, :, 1] = (pred[:, :, 1] - (pred[:, :, 3] / 2)) # top-left_y
    bbox_corner[:, :, 2] = (pred[:, :, 0] + (pred[:, :, 2] / 2)) # bottom_right_x
    bbox_corner[:, :, 3] = (pred[:, :, 1] + (pred[:, :, 3] / 2)) # bottom_right_y
    pred[:, :, :4] = bbox_corner[:, :, :4]

    # each image in the batch will have varying numbers of true detections
    output = None
    for idx in range(pred.shape[0]):
        img_pred = pred[idx]

        # pick the class with maximum score, add the score and the index
        # to the prediction.
        max_conf, max_idx = torch.max(img_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1).to(device)
        max_idx = max_idx.float().unsqueeze(1).to(device)
        img_pred = torch.cat([img_pred[:, :5], max_conf, max_idx], 1)

        non_zero_idx = torch.nonzero(img_pred[:, 4]).to(device)
        img_pred = img_pred[non_zero_idx.squeeze(), :].view(-1, 7).to(device)
        if not img_pred.shape[0]:
            continue

        # get the unique classes detected in the image.
        img_classes = torch.unique(img_pred[:, -1]).to(device)
        
        # for each object in the image, get the one true bounding box that
        # contains the object.
        for cls in img_classes:
            class_mask = img_pred * (img_pred[:, -1] == cls).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(class_mask[:, -2]).squeeze()
            img_pred_class = img_pred[class_mask_idx].view(-1, 7)

            # sort the detections in decreasing order of the objectness score
            conf_sort_idx = torch.sort(img_pred_class[:, 4], descending=True)[1]
            img_pred_class = img_pred_class[conf_sort_idx]

            # since the bounding boxes have been sorted in decreasing order of the
            # objectness score, pick the one with the maximum objectness score and
            # use non-maximum suppression to remove all other boxes that might be
            # detecting the same object as the one with max objectness score.
            for d_idx in range(img_pred_class.shape[0]):
                try:
                    ious = bbox_iou(img_pred_class[d_idx].unsqueeze(0), img_pred_class[d_idx+1:], device=device)
                except (IndexError, ValueError):
                    break

                # remove overlapping bounding boxes
                iou_mask = (ious < nms_confidence_level).float().unsqueeze(1)
                img_pred_class[d_idx+1:] *= iou_mask
                non_zero_idx = torch.nonzero(img_pred_class[:, 4]).squeeze()
                img_pred_class = img_pred_class[non_zero_idx].view(-1, 7)

            batch_idx = img_pred_class.new(img_pred_class.shape[0], 1).fill_(idx)
            if isinstance(output, torch.Tensor):
                out = torch.cat([batch_idx, img_pred_class], 1)
                output  = torch.cat([output, out])
            else:
                output = torch.cat([batch_idx, img_pred_class], 1)
    return output

