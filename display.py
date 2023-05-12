from os import makedirs
from pathlib import Path
from PIL import ImageDraw, Image
from torch import Tensor
import colorsys
import math
import matplotlib.pyplot as pt


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
    preds = preds.int().tolist()
    return preds


def print_prediction(pred: Tensor, label):
    text = f"{label} {pred[4]/100}"
    position = f"{pred[0]:5}, {pred[1]:5}, {pred[2]:5}, {pred[3]:5}"
    print(f"- {text} ({position})")


C_WHITE = (255, 255, 255)
C_CLASS = lambda id: tuple(
    [int(round(v * 255)) for v in colorsys.hsv_to_rgb(id * math.e % math.tau, 1, 0.8)]
)


# Draw a single prediction onto the image
def draw_prediction(draw: ImageDraw.ImageDraw, pred, label):
    text = f"{label} {pred[4]/100}"
    (tx, ty) = draw.textsize(text)
    rect_color = C_CLASS(pred[5])
    draw.rectangle((pred[0], pred[1], pred[0] + tx, pred[1] + ty), rect_color)
    draw.rectangle([(pred[0], pred[1]), (pred[2], pred[3])], outline=rect_color)
    draw.text((pred[0], pred[1]), text, C_WHITE)


# Process multiple batches of predictions
# Letterboxing not supported
def process_results(
    results,
    out_size,
    console=False,
    show=False,
    out_dir=None,
    labels=None,
):
    if out_dir is not None:
        makedirs(out_dir)
    for batch in results:
        for path, predictions, raw_image in zip(*batch):
            ow, oh = raw_image.width, raw_image.height
            om = max(ow, oh)
            ratio = om / out_size
            nw, nh = int(round(ow / ratio)), int(round(oh / ratio))
            draw = ImageDraw.Draw(raw_image)
            if console:
                print(f"Image `{path}`")
            for pred in predictions:
                pred = clamp_rect(pred.clone(), (0, 0, 415, 415))
                pred = unpad(pred, ((out_size - nw) // 2, (out_size - nh) // 2))
                pred = scale(pred, (ratio, ratio))
                pred = finalize_predictions(pred)
                label = labels[pred[5]] if labels else str(pred[5] + 1)
                if console:
                    print_prediction(pred, label)
                if out_dir is not None or show:
                    draw_prediction(draw, pred, label)
            if out_dir is not None:
                raw_image.save(out_dir.joinpath(path))
            if show:
                raw_image.show()


def display_dir(path: Path):
    images = []
    for file in path.iterdir():
        if file.suffix == ".jpg":
            images.append(file)
    fig, ax = pt.subplots(len(images))
    for i, file in enumerate(images):
        image = Image.open(file)
        ax[i].imshow(image)
    pt.show()
