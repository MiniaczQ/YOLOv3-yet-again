from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
from torchvision.ops import nms, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# Squares off the image with padding
class PadToSquare:
    def __init__(self, value):
        self.value = value

    def __call__(self, t: Tensor) -> Tensor:
        _, h, w = t.shape
        m = max(h, w)

        dw = m - w
        lp = dw // 2
        rp = dw - lp

        dh = m - h
        tp = dh // 2
        bp = dh - tp

        return F.pad(t, (lp, rp, tp, bp), value=self.value)


# Downsizes the image to fit inside max_size, while keeping the aspect ratio
class ResizeKeepRatio:
    def __init__(self, max_size, antialias=False):
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, t: Tensor) -> Tensor:
        _, h, w = t.shape
        ratio = self.max_size / max(h, w)

        h = int(round(h * ratio))
        w = int(round(w * ratio))
        return TF.resize(
            t, [h, w], TF.InterpolationMode.BILINEAR, antialias=self.antialias
        )


# Scale bboxes in annotations
class NormalizeBbox:
    def __init__(self, image_size: tuple[int, int], padded=True):
        self.image_size = image_size
        self.padded = padded

    @staticmethod
    def _map_linearly(
        x: torch.Tensor, in_range: tuple[float, float], out_range: tuple[float, float]
    ):
        assert in_range[1] > in_range[0]
        assert out_range[1] > out_range[0]
        scale = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
        return (x - in_range[0]) * scale + out_range[0]

    def __call__(self, annotation: torch.Tensor) -> torch.Tensor:
        annotation = annotation.clone().float()
        sizes = annotation[..., 3:5] - annotation[..., 1:3]
        annotation[..., 1:] = torch.cat((annotation[..., 1:3] + sizes / 2, sizes), 1)
        max_size = max(self.image_size)
        size_diff = (max_size - min(self.image_size)) / 2
        if self.image_size[0] > self.image_size[1]:
            padding = torch.tensor([0, size_diff])
        else:
            padding = torch.tensor([size_diff, 0])
        annotation[..., 1:3] = annotation[..., 1:3] + padding
        annotation[..., 1:] = annotation[..., 1:] / max_size
        return annotation


# Based on https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
# Normalize a batch of predictions from 1 detector
def normalize_model_output(bpreds: Tensor, num_anchors: int, bbox_attrs: int) -> Tensor:
    batch_size = bpreds.size(0)
    pred_dim = bpreds.size(2)

    # B x A*(5+N) x H x W
    bpreds = bpreds.view(batch_size, num_anchors, bbox_attrs, pred_dim, pred_dim)
    # B x A x (5+N) x H x W
    bpreds = bpreds.permute(0, 1, 3, 4, 2)
    # B x A x H x W x (5+N)

    # Sigmoid offsets
    bpreds[..., [0, 1]] = torch.sigmoid(bpreds[..., [0, 1]])
    # Sigmoid object confidence and class probabilities
    bpreds[..., 4:] = torch.sigmoid(bpreds[..., 4:])

    return bpreds


# Based on
# https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
# https://github.com/Lornatang/YOLOv3-PyTorch/tree/main
# Not so shrimple
# Process a batch of predictions from 1 detector
def process_anchor(
    bpreds: Tensor, inp_dim: int, anchors: Tensor, num_classes: int
) -> Tensor:
    num_anchors = anchors.size(0)
    batch_size = bpreds.size(0)
    pred_dim = bpreds.size(2)
    bbox_attrs = 5 + num_classes
    stride = inp_dim // pred_dim
    # B x A*(5+N) x H x W
    bpreds = bpreds.view(batch_size, num_anchors, bbox_attrs, pred_dim, pred_dim)
    # B x A x (5+N) x H x W
    bpreds = bpreds.permute(0, 1, 3, 4, 2)
    # B x A x H x W x (5+N)
    grid_axis = torch.arange(pred_dim, dtype=torch.float32, device=bpreds.device)
    grid = torch.cartesian_prod(grid_axis, grid_axis).view(1, 1, pred_dim, pred_dim, 2)[
        ..., [1, 0]
    ]
    xy = (torch.sigmoid(bpreds[..., [0, 1]]) + grid) * stride
    wh = torch.exp(bpreds[..., [2, 3]]) * anchors.view(1, num_anchors, 1, 1, 2)
    attr = torch.sigmoid(bpreds[..., 4:])
    return torch.cat([xy, wh, attr], 4).view(batch_size, -1, bbox_attrs)


def process_output_without_sigmoid(
    bpreds: Tensor, inp_dim: int, anchors: Tensor, num_classes: int
) -> Tensor:
    num_anchors = anchors.size(0)
    batch_size = bpreds.size(0)
    pred_dim = bpreds.size(2)
    bbox_attrs = 5 + num_classes
    stride = inp_dim // pred_dim
    # B x A x H x W x (5+N)
    grid_axis = torch.arange(pred_dim, dtype=torch.float32, device=bpreds.device)
    grid = torch.cartesian_prod(grid_axis, grid_axis).view(1, 1, pred_dim, pred_dim, 2)[
        ..., [1, 0]
    ]
    xy = (bpreds[..., [0, 1]] + grid) * stride
    wh = torch.exp(bpreds[..., [2, 3]]) * anchors.view(1, num_anchors, 1, 1, 2)
    attr = bpreds[..., 4:]
    return torch.cat([xy, wh, attr], 4).view(batch_size, -1, bbox_attrs)


# Filter away predictions with low objectness score
def threshold_objectness(preds: Tensor, oc_threshold: float) -> Tensor:
    mask = preds[:, 4] > oc_threshold
    return preds[mask, :]


# Keep only the most probably class
def keep_best_class(preds: Tensor) -> Tensor:
    class_id, class_prob = torch.max(preds[:, 5:], dim=1)
    preds = preds[..., :5]
    class_id = class_id.float().view(-1, 1)
    class_prob = class_prob.float().view(-1, 1)
    preds = torch.cat([preds, class_id, class_prob], dim=1)
    return preds


# Convert center_x, center_y, width, height to rectangles
def xywh_to_rect(xywhs: Tensor):
    rects = xywhs.clone()
    rects[..., [2, 3]] /= 2
    rects[..., [0, 1]] = xywhs[..., [0, 1]] - rects[..., [2, 3]]
    rects[..., [2, 3]] = xywhs[..., [0, 1]] + rects[..., [2, 3]]
    return rects


# Confirmed working with another model
# Non max suppresion
def non_max_supression(bpreds: Tensor, conf_threshold, size_limits, iou_threshold):
    # Run postprocessing and non max suppression on every image's predictions
    results = []
    for preds in bpreds:
        # Filter away low objectivness predictions
        preds = preds[preds[:, 4] > conf_threshold]
        # Filter away invalid prediction sizes
        min_mask = preds[:, [2, 3]] > size_limits[0]
        max_mask = preds[:, [2, 3]] < size_limits[1]
        preds = preds[(min_mask & max_mask).all(1)]
        # Early return
        if preds.size(0) == 0:
            results.append(preds)
            continue
        # Convert to rectangles
        boxes = xywh_to_rect(preds[:, [0, 1, 2, 3]])
        # Best prediction for each box
        confs, ids = preds[:, 5:].max(1)
        # Include objectness in class probability
        confs *= preds[:, 4]
        # Reduce predictions to x1, y1, x2, y2, class probability, class id
        preds = torch.cat([boxes, confs.view(-1, 1), ids.view(-1, 1)], 1)
        # Filter away low probability classes
        preds = preds[confs > conf_threshold]
        # Early return
        num_boxes = preds.size(0)
        if num_boxes == 0:
            results.append(preds)
            continue
        # Batched nms
        classes = preds[:, 5]  # classes
        boxes = preds[:, :4].clone() + classes.view(-1, 1) * size_limits[1]
        scores = preds[:, 4]
        idxs = nms(boxes, scores, iou_threshold)
        # Box merging using weighted mean
        if 1 < num_boxes:
            # For each nms selected box, find all other boxes that are overlapping
            iou = box_iou(boxes[idxs], boxes) > iou_threshold
            # Assign weight to each box
            weights = iou * scores.view(1, -1)
            # Perform a weighted mean
            preds[idxs, :4] = torch.mm(weights, preds[:, :4]).float() / weights.sum(
                1, keepdim=True
            )
        results.append(preds[idxs])
    return results


# Predictions as:
# batch[image[prediction[x1, y1, x2, y2, class, confidence]]]
# Targets as:
# batch[prediction[img_id, class, x, y, w, h]]
# Because no time to refactor
def calculate_map(batch_preds: list, batch_targs: torch.tensor) -> dict:
    map = MeanAveragePrecision()
    # Process targets to the same format as predictions
    batch_targs = xywh_to_rect(batch_targs[:, [2, 3, 4, 5, 1, 0]])
    batch_targs[:, [0, 1, 2, 3]] *= 416
    batch_img_targs = []
    for img_id in range(len(batch_preds)):
        mask = batch_targs[:, 5] == img_id
        batch_img_targs.append(batch_targs[mask, :])
    # Update mAP for each image in batch
    update_preds = []
    update_targs = []
    for targets, predictions in zip(batch_img_targs, batch_preds):
        update_preds.append(
            dict(
                boxes=predictions[:, [0, 1, 2, 3]],
                scores=predictions[:, 4],
                labels=predictions[:, 5],
            )
        )
        update_targs.append(
            dict(
                boxes=targets[:, [0, 1, 2, 3]],
                labels=targets[:, 4],
            )
        )
    map.update(update_preds, update_targs)
    return map.compute()
