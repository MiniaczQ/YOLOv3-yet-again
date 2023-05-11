from os import makedirs
from datetime import datetime
from pathlib import Path
from PIL import ImageDraw
import torch
from torch import Tensor
import colorsys
import math


# Clamp rectangles in another rectangle
def clamp_rect(rects: Tensor, rect):
    rects[..., [0, 2]].clamp_(rect[0], rect[2])
    rects[..., [1, 3]].clamp_(rect[1], rect[3])
    return rects


# Scale rectangles
def scale(rects: Tensor, scale):
    rects[..., [0, 2]] *= scale[0]
    rects[..., [1, 3]] *= scale[1]
    return rects


# Remove padding (left & top)
def unpad(rects: Tensor, padding):
    rects[..., [0, 2]] -= padding[0]
    rects[..., [1, 3]] -= padding[1]
    return rects


# Represent confidence in percentage and turn predictions into integer lists
def finalize_predictions(preds: Tensor):
    preds[..., 4] *= 100
    preds[..., 5] += 1
    preds = preds.int().tolist()
    return preds


def print_prediction(p: Tensor):
    print(f"{p[5]:5} {p[4]/100} ({p[0]:5}, {p[1]:5}, {p[2]:5}, {p[3]:5})")


C_WHITE = (255, 255, 255)
C_CLASS = lambda id: tuple(
    [int(round(v * 255)) for v in colorsys.hsv_to_rgb(id * math.e % math.tau, 1, 0.8)]
)


# Draw a single prediction onto the image
def draw_prediction(draw: ImageDraw.ImageDraw, pred):
    text = f"{pred[5]}:{pred[4]/100}"
    (tx, ty) = draw.textsize(text)
    rect_color = C_CLASS(pred[5])
    draw.rectangle((pred[0], pred[1], pred[0] + tx, pred[1] + ty), rect_color)
    draw.rectangle([(pred[0], pred[1]), (pred[2], pred[3])], outline=rect_color)
    draw.text((pred[0], pred[1]), text, C_WHITE)


# confirmed with another model as long as model input was 416x416 (no letterboxing)
def show_results(results, out_size):
    results_dir = Path(datetime.now().strftime("detection_results/%Y_%m_%d_%H_%M_%S"))
    makedirs(results_dir)
    for batch in results:
        for path, predictions, raw_image in zip(*batch):
            ow, oh = raw_image.width, raw_image.height
            om = max(ow, oh)
            ratio = om / out_size
            nw, nh = int(round(ow / ratio)), int(round(oh / ratio))
            draw = ImageDraw.Draw(raw_image)
            for pred in predictions:
                pred = clamp_rect(pred.clone(), (0, 0, 415, 415))
                pred = unpad(pred, ((out_size - nw) // 2, (out_size - nh) // 2))
                pred = scale(pred, (ratio, ratio))
                pred = finalize_predictions(pred)
                draw_prediction(draw, pred)
            raw_image.save(results_dir.joinpath(path))
