from torch import Tensor
import torch.nn.functional as F
import torch
from torchvision import ops


# Squares off the image with padding
def square_padding(t: Tensor):
    _, h, w = t.shape
    m = max(h, w)

    dw = m - w
    lp = dw // 2
    rp = dw - lp

    dh = m - h
    tp = dh // 2
    bp = dh - tp

    return F.pad(t, (lp, rp, tp, bp))


# Unsqueeze
def unsqueeze_dim0(t: Tensor):
    return t.unsqueeze(0)


# Based on https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
# Process a batch of predictions from 1 detector
def process_prediction(
    bpreds: Tensor, inp_dim: int, anchors: Tensor, num_classes: int, device=None
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
    bpreds[:, :, :, :, 4:] = torch.sigmoid(bpreds[:, :, :, :, 4:])

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
def process_predictions(
    bpreds, input_size: int, anchors: Tensor, num_classes: int, device=None
):
    (bx52, bx26, bx13) = bpreds
    bx52 = process_prediction(bx52, input_size, anchors[[0, 1, 2]], num_classes, device)
    bx26 = process_prediction(bx26, input_size, anchors[[3, 4, 5]], num_classes, device)
    bx13 = process_prediction(bx13, input_size, anchors[[6, 7, 8]], num_classes, device)
    return torch.cat([bx52, bx26, bx13], dim=1)


# Filter away predictions with low object score
def threshold_object_confidence(preds: Tensor, oc_threshold: float):
    mask = preds[:, 4] > oc_threshold
    return preds[mask, :]


# Turn predictions into AABB with class index
def process_with_nms(preds: Tensor, num_classes: int, iou_threshold: float):
    boxes = preds[:, [0, 1, 2, 3]]
    correct_boxes = []
    for class_id in torch.arange(num_classes):
        scores = preds[:, 5 + class_id]
        curr_keep_indices = ops.nms(boxes, scores, iou_threshold)
        num_kept = len(curr_keep_indices)
        kept_boxes = boxes[curr_keep_indices, :]
        kept_scores = scores[curr_keep_indices].view(-1, 1)
        class_ids = (
            torch.tensor(class_id, device=preds.device).view(1, 1).repeat(num_kept, 1)
        )
        correct_boxes.append(torch.cat([kept_boxes, class_ids, kept_scores], 1))
    return torch.cat(correct_boxes, 0)


# Processes a batch of predictions into a batch of bounding boxes with class indices
def process_into_aabbs(
    bpreds,
    input_size: int,
    anchors: Tensor,
    num_classes: int,
    oc_threshold: float,
    iou_threshold: float,
    device=None,
):
    bpreds = process_predictions(bpreds, input_size, anchors, num_classes, device)
    print(bpreds.shape)
    for preds in bpreds:
        print()
        print(preds.shape)
        preds = threshold_object_confidence(preds, oc_threshold)
        print(preds.shape)
        preds = process_with_nms(preds, num_classes, iou_threshold)
        print(preds.shape)
        yield preds
