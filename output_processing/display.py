import colorsys
from os import makedirs
import math
from PIL import ImageDraw
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as pt
import matplotlib as mpl
import torch
from torch import Tensor


# Clamp rectangles within another rectangle
def clamp_rect(rects: Tensor, rect):
    rects[..., [0, 2]] = rects[..., [0, 2]].clamp(rect[0], rect[2])
    rects[..., [1, 3]] = rects[..., [1, 3]].clamp(rect[1], rect[3])
    return rects


# Scale rectangles
def scale(rects: Tensor, scale):
    rects[..., [0, 2]] *= scale[0]
    rects[..., [1, 3]] *= scale[1]
    return rects


# Remove padding from rectangles (translation)
def unpad(rects: Tensor, padding):
    rects[..., [0, 2]] -= padding[0]
    rects[..., [1, 3]] -= padding[1]
    return rects


# Represent confidence in percentage and turn predictions into integer lists
def finalize_predictions(preds: Tensor) -> list[int]:
    preds[..., 4] *= 100
    return preds.int().tolist()


# Print a prediction to console
def print_prediction(pred: Tensor, label):
    text = f"{label} {pred[4]/100:4.2}"
    position = f"{pred[0]:5}, {pred[1]:5}, {pred[2]:5}, {pred[3]:5}"
    print(f"- {text} ({position})")


# Text color
C_WHITE = (255, 255, 255)


# Id-based class color using a hash function to determine hue
# hash(input) = input * K mod W
# where K = phi (golden ratio)
# and W = 2 pi
phi = (1 + math.sqrt(5)) / 2
C_CLASS = lambda id: tuple(
    [int(round(v * 255)) for v in colorsys.hsv_to_rgb(id * phi % math.tau, 1, 0.8)]
)


# Draw a single prediction onto the image
def draw_prediction(draw: ImageDraw.ImageDraw, pred: list[int], label):
    text = f"{label} {pred[4]/100:4.2}"
    (tx, ty) = draw.textsize(text)
    rect_color = C_CLASS(pred[5])
    draw.rectangle((pred[0], pred[1], pred[0] + tx, pred[1] + ty), rect_color)
    draw.rectangle(((pred[0], pred[1]), (pred[2], pred[3])), outline=rect_color)
    draw.text((pred[0], pred[1]), text, C_WHITE)


# Add image to display queue
def add_image(image):
    pt.figure()
    pt.axis("off")
    pt.tight_layout()
    pt.imshow(image)


# Process multiple batches of predictions
def output_results(
    results,
    out_size,
    console=False,
    show_n=0,
    out_dir=None,
    labels=None,
    flatten_output=True,
):
    with torch.no_grad():
        # Process arguments
        show = False
        if out_dir is not None:
            out_dir = Path(out_dir)
            if out_dir.exists():
                rmtree(out_dir)
        if labels is not None:
            label_pad = max([len(l) for l in labels])
        else:
            label_pad = math.ceil(
                math.log10(
                    max([max([max(preds[5]) for preds in batch]) for batch in results])
                )
            )
        if show_n != 0:
            mpl.rcParams["figure.max_open_warning"] = 0
            show = True

        # Iterate over each image
        for batch in results:
            for path, predictions, raw_image in zip(*batch):
                if flatten_output:
                    path = "__".join(Path(path).parts)
                # Calculate padding of the image so we can reverse it
                ow, oh = raw_image.width, raw_image.height
                om = max(ow, oh)
                ratio = om / out_size
                nw, nh = int(round(ow / ratio)), int(round(oh / ratio))

                draw = ImageDraw.Draw(raw_image)
                if console:
                    print(f"Image `{path}`")
                # Process each box
                for pred in predictions:
                    pred = unpad(
                        pred.clone(), ((out_size - nw) // 2, (out_size - nh) // 2)
                    )
                    pred = scale(pred, (ratio, ratio))
                    pred = clamp_rect(pred, (0, 0, ow - 1, oh - 1))
                    pred = finalize_predictions(pred)

                    # Print & draw the processed detection
                    label = labels[pred[5]] if labels else str(pred[5] + 1)
                    if console:
                        print_prediction(pred, f"{label:{label_pad}}")
                    if out_dir is not None or show_n:
                        draw_prediction(draw, pred, label)

                if out_dir is not None:
                    save_path = out_dir.joinpath(path)
                    makedirs(save_path.parent, exist_ok=True)
                    raw_image.save(save_path)

                if show_n > 0:
                    add_image(raw_image)
                    show_n -= 1

        if show is not None and show:
            pt.show()
