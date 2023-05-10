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


# Scale bboxes in annotations
def normalize_bbox(image_size: tuple[int, int], padded=True):
    def _map_linearly(
        x: torch.Tensor, in_range: tuple[float, float], out_range: tuple[float, float]
    ):
        assert in_range[1] > in_range[0]
        assert out_range[1] > out_range[0]
        scale = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
        return (x - in_range[0]) * scale + out_range[0]

    def _normalize_bbox(annotation: torch.Tensor) -> torch.Tensor:
        annotation = annotation.clone().float()
        ratio = image_size[0] / image_size[1]
        annotation[..., [1, 3]] = _map_linearly(
            annotation[..., [1, 3]], (0, image_size[0] - 1), (0, 1)
        )
        annotation[..., [2, 4]] = _map_linearly(
            annotation[..., [2, 4]], (0, image_size[1] - 1), (0, 1)
        )
        if padded and ratio < 1:
            annotation[..., 1] = _map_linearly(
                annotation[..., 1], (0, 1), ((1 - ratio) / 2, (1 + ratio) / 2)
            )
            annotation[..., 3] = _map_linearly(annotation[..., 3], (0, 1), (0, ratio))
        if padded and ratio > 1:
            inv_ratio = 1 / ratio
            annotation[..., 2] = _map_linearly(
                annotation[..., 2], (0, 1), ((1 - inv_ratio) / 2, (1 + inv_ratio) / 2)
            )
            annotation[..., 4] = _map_linearly(
                annotation[..., 4], (0, 1), (0, inv_ratio)
            )
        return annotation

    return _normalize_bbox


# Unsqueeze
def unsqueeze_dim0(t: Tensor):
    return t.unsqueeze(0)


# Based on https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
# Process a batch of predictions from 1 detector
def process_prediction(bpreds: Tensor, inp_dim: int, anchors: Tensor, num_classes: int):
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
    bpreds[..., [0, 1]] = torch.sigmoid(bpreds[..., [0, 1]])
    # Sigmoid object confidence and class probabilities
    bpreds[..., 4:] = torch.sigmoid(bpreds[..., 4:])

    # Offsets 1 x 1 x W x H x 2
    grid = torch.arange(pred_dim, dtype=torch.float32, device=bpreds.device)
    xy_offsets = torch.cartesian_prod(grid, grid).view(1, 1, pred_dim, pred_dim, 2)

    # Add offsets
    bpreds[..., [0, 1]] += xy_offsets

    # Scale 1 x A x 1 x 1 x 2
    anchors = anchors.view(1, num_anchors, 1, 1, 2)

    # Calculate size
    bpreds[..., [2, 3]] = torch.exp(bpreds[..., [2, 3]])
    bpreds[..., [2, 3]] *= anchors

    # Upscale
    bpreds[..., [0, 1, 2, 3]] *= scale

    # Calculate coordinates
    bpreds[..., [2, 3]] += bpreds[..., [1, 2]]

    # If we want a perfect result match with the tutorial, we need to use the exact same order before reducing dimensions
    # In practice, all anchor, width and height relevant information are already inside the attributes, so we don't need to keep them in any specific order
    # preds = preds.permute(0, 3, 2, 1, 4)

    # Reduce dimensions
    bpreds = bpreds.reshape(batch_size, -1, bbox_attrs)
    return bpreds


# Process a batch of predictions from all 3 detectors
def process_predictions(bpreds, input_size: int, anchors: Tensor, num_classes: int):
    (bx52, bx26, bx13) = bpreds
    bx52 = process_prediction(bx52, input_size, anchors[[0, 1, 2]], num_classes)
    bx26 = process_prediction(bx26, input_size, anchors[[3, 4, 5]], num_classes)
    bx13 = process_prediction(bx13, input_size, anchors[[6, 7, 8]], num_classes)
    return torch.cat([bx52, bx26, bx13], dim=1)


# Filter away predictions with low objectness score
def threshold_objectness(preds: Tensor, oc_threshold: float):
    mask = preds[:, 4] > oc_threshold
    return preds[mask, :]


# Keep only the most probably class
def keep_best_class(preds: Tensor):
    class_id, class_prob = torch.max(preds[:, 5:], dim=1)
    preds = preds[..., :5]
    class_id = class_id.float().view(-1, 1)
    class_prob = class_prob.float().view(-1, 1)
    preds = torch.cat([preds, class_id, class_prob], dim=1)
    return preds


# Turn predictions into AABB with class index
def batched_nms(preds: Tensor, iou_threshold: float):
    boxes = preds[:, [0, 1, 2, 3]]
    scores = preds[:, 5]
    idxs = preds[:, 6]
    kept_idxs = ops.batched_nms(boxes, scores, idxs, iou_threshold)
    preds = preds[kept_idxs]
    return preds


# Processes a batch of predictions into a batch of bounding boxes with class indices
def process_into_aabbs(
    bpreds,
    input_size: int,
    anchors: Tensor,
    num_classes: int,
    oc_threshold: float,
    iou_threshold: float,
):
    results = []
    bpreds = process_predictions(bpreds, input_size, anchors, num_classes)
    for preds in bpreds:
        preds = threshold_objectness(preds, oc_threshold)
        preds = keep_best_class(preds)
        preds = batched_nms(preds, iou_threshold)
        results.append(preds)
    return results
