from PIL import Image
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms


def pad(t: Tensor):
    _, h, w = t.shape
    m = max(h, w)

    dw = m - w
    lp = dw // 2
    rp = dw - lp

    dh = m - h
    tp = dh // 2
    bp = dh - tp

    t = F.pad(t, (lp, rp, tp, bp))
    return t


def resize(t: Tensor):
    t = F.interpolate(t.unsqueeze(0), (416, 416), mode="nearest").squeeze()
    return t


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(pad),
        transforms.Lambda(resize),
    ]
)


def get_img(path):
    return transform(Image.open(path).convert("RGB")).cuda().unsqueeze(0)
