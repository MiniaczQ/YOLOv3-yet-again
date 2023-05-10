from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
from torchvision import ops


# Squares off the image with padding
class PadToSquare:
    def __call__(self, t: Tensor) -> Tensor:
        _, h, w = t.shape
        m = max(h, w)

        dw = m - w
        lp = dw // 2
        rp = dw - lp

        dh = m - h
        tp = dh // 2
        bp = dh - tp

        return F.pad(t, (lp, rp, tp, bp))


# Downsizes the image to fit inside max_size, while keeping the aspect ratio
class ResizeKeepRatio:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, t: Tensor) -> Tensor:
        _, h, w = t.shape
        ratio = self.max_size / max(h, w)

        h = int(round(h * ratio))
        w = int(round(w * ratio))
        return TF.resize(t, [h, w], TF.InterpolationMode.BILINEAR)


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
        ratio = self.image_size[0] / self.image_size[1]
        annotation[..., [1, 3]] = self._map_linearly(
            annotation[..., [1, 3]], (0, self.image_size[0] - 1), (0, 1)
        )
        annotation[..., [2, 4]] = self._map_linearly(
            annotation[..., [2, 4]], (0, self.image_size[1] - 1), (0, 1)
        )
        if self.padded and ratio < 1:
            annotation[..., 1] = self._map_linearly(
                annotation[..., 1], (0, 1), ((1 - ratio) / 2, (1 + ratio) / 2)
            )
            annotation[..., 3] = self._map_linearly(
                annotation[..., 3], (0, 1), (0, ratio)
            )
        if self.padded and ratio > 1:
            inv_ratio = 1 / ratio
            annotation[..., 2] = self._map_linearly(
                annotation[..., 2], (0, 1), ((1 - inv_ratio) / 2, (1 + inv_ratio) / 2)
            )
            annotation[..., 4] = self._map_linearly(
                annotation[..., 4], (0, 1), (0, inv_ratio)
            )
        return annotation


# Unsqueeze
def unsqueeze_dim0(t: Tensor) -> Tensor:
    return t.unsqueeze(0)


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
# Process a batch of predictions from 1 detector
def process_anchor(
    bpreds: Tensor, inp_dim: int, anchors: Tensor, num_classes: int
) -> Tensor:
    num_anchors = anchors.size(0)
    batch_size = bpreds.size(0)
    pred_dim = bpreds.size(2)
    bbox_attrs = 5 + num_classes

    # Sigmoid & transform to B x A x H x W x (5+N) representation
    bpreds = normalize_model_output(bpreds, num_anchors, bbox_attrs)

    # Position
    grid_axis = torch.arange(pred_dim, dtype=torch.float32, device=bpreds.device)
    grid = torch.cartesian_prod(grid_axis, grid_axis).view(1, 1, pred_dim, pred_dim, 2)
    bpreds[..., [0, 1]] += grid

    # Size
    scale = inp_dim // pred_dim
    anchors /= scale
    anchors = anchors.view(1, num_anchors, 1, 1, 2)
    bpreds[..., [2, 3]] = torch.exp(bpreds[..., [2, 3]]) * anchors

    # Upscale
    bpreds[..., [0, 1, 2, 3]] *= scale

    # If we want a perfect result match with the tutorial, we need to use the exact same order before reducing dimensions
    # In practice, all anchor, width and height relevant information are already inside the attributes, so we don't need to keep them in any specific order
    # preds = preds.permute(0, 3, 2, 1, 4)

    # Reduce dimensions
    bpreds = bpreds.reshape(batch_size, -1, bbox_attrs)
    return bpreds


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


# Turn predictions into AABB with class index
def batched_nms(preds: Tensor, iou_threshold: float) -> Tensor:
    boxes = preds[:, [0, 1, 2, 3]]
    scores = preds[:, 5]
    idxs = preds[:, 6]
    kept_idxs = ops.batched_nms(boxes, scores, idxs, iou_threshold)
    preds = preds[kept_idxs]
    return preds


# Convert center_x, center_y, width, height to rectangles
def xywh_to_rect(xywhs: Tensor):
    rects = xywhs.clone()
    rects[..., [2, 3]] /= 2
    rects[..., [0, 1]] = xywhs[..., [0, 1]] - rects[..., [2, 3]]
    rects[..., [2, 3]] = xywhs[..., [0, 1]] + rects[..., [2, 3]]
    return rects
