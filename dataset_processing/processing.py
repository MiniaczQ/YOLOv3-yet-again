import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF


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
