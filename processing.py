from torch import Tensor
import torch.nn.functional as F
import torch
from torchvision import ops


# Squares off the image with padding
def pad(t: Tensor):
    _, h, w = t.shape
    m = max(h, w)

    dw = m - w
    lp = dw // 2
    rp = dw - lp

    dh = m - h
    tp = dh // 2
    bp = dh - tp

    return F.pad(t, (lp, rp, tp, bp))


# Resizes thhe image to 416x416
def resize(t: Tensor):
    return F.interpolate(t.unsqueeze(0), (416, 416), mode="bilinear").squeeze()


# Based on https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
# Process a batch of predictions from 1 detector
def process_prediction(
    bpreds: Tensor, inp_dim, anchors: Tensor, num_classes, device=None
):
    num_anchors = anchors.size(0)
    batch_size = bpreds.size(0)
    pred_dim = bpreds.size(2)
    bbox_attrs = 5 + num_classes
    scale = inp_dim // pred_dim

    # B x A*(5+N) x W x H
    bpreds = bpreds.view(batch_size, num_anchors, bbox_attrs, pred_dim, pred_dim)
    # B x A x (5+N) x W x H
    bpreds = bpreds.permute(0, 1, 3, 4, 2)
    # B x A x W x H x (5+N)

    # Scale down anchors
    anchors /= scale

    # Sigmoid offsets
    bpreds[:, :, :, :, [0, 1]] = torch.sigmoid(bpreds[:, :, :, :, [0, 1]])
    # Sigmoid object confidence and class probabilities
    bpreds[:, :, :, :, 4:] = torch.sigmoid(bpreds[:, :, :, :, 4])

    # Offsets 1 x 1 x W x H x 2
    grid = torch.arange(pred_dim, dtype=torch.float32, device=device)
    xy_offsets = torch.cartesian_prod(grid, grid).view(1, 1, pred_dim, pred_dim, 2)

    # Add offsets
    bpreds[:, :, :, :, [0, 1]] += xy_offsets

    # Scale 1 x A x 1 x 1 x 2
    anchors = anchors.view(1, num_anchors, 1, 1, 2)

    # Calculate size
    bpreds[:, :, :, :, [2, 3]] = torch.exp(bpreds[:, :, :, :, [2, 3]])
    bpreds[:, :, :, :, [2, 3]] *= anchors

    # Upscale
    bpreds[:, :, :, :, [0, 1, 2, 3]] *= scale

    # If we want a perfect result match with the tutorial, we need to use the exact same order before reducing dimensions
    # In practice, all anchor, width and height relevant information are already inside the attributes, so we don't need to keep them in any specific order
    # preds = preds.permute(0, 3, 2, 1, 4)

    # Reduce dimensions
    bpreds = bpreds.reshape(batch_size, -1, bbox_attrs)
    return bpreds


# Process a batch of predictions from all 3 detectors
def process_predictions(bpreds, input_size, anchors, num_classes, device=None):
    (bx52, bx26, bx13) = bpreds
    bx52 = process_prediction(bx52, input_size, anchors[[0, 1, 2]], num_classes, device)
    bx26 = process_prediction(bx26, input_size, anchors[[3, 4, 5]], num_classes, device)
    bx13 = process_prediction(bx13, input_size, anchors[[6, 7, 8]], num_classes, device)
    return torch.cat([bx52, bx26, bx13], dim=1)


# Filter away predictions with low object score
def threshold_object_confidence(preds: Tensor, oc_threshold):
    mask = preds[:, 4] > oc_threshold
    return preds[mask, :]


# Turn predictions into AABB with class index
def process_with_nms(preds: Tensor, iou_threshold):
    classes = torch.argmax(preds[:, 5:], dim=1)
    indices = ops.nms(preds[:, [0, 1, 2, 3]], preds[:, 5:], iou_threshold)
    boxes = preds[indices, [0, 1, 2, 3]]
    return torch.stack([boxes, classes], dim=1)


# Processes a batch of predictions into a batch of bounding boxes with class indices
def process_into_aabbs(
    bpreds, input_size, anchors, num_classes, oc_threshold, iou_threshold, device=None
):
    bpreds = process_predictions(bpreds, input_size, anchors, num_classes, device)
    for preds in bpreds:
        preds = threshold_object_confidence(preds, oc_threshold)
        preds = process_with_nms(preds, iou_threshold)
        yield preds
