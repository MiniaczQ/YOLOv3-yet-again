from lightning import Trainer
import torch
from datamodule import Datamodule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from simple_dataset import SimpleDataset

from yolov3 import YoloV3Module


from PIL import Image, ImageDraw


from torchvision import transforms
from os import mkdir
from datetime import datetime
from pathlib import Path


def show_results(ds, batch_size, bsaabbs):
    results_dir = Path(datetime.now().strftime("results_%Y_%m_%d_%H_%M_%S"))
    mkdir(results_dir)
    for i, baabbs in enumerate(bsaabbs):
        for j, aabbs in enumerate(baabbs):
            path, img = ds.get_raw(batch_size * i + j)
            w, h = img.width, img.height
            draw = ImageDraw.Draw(img)
            print(f"Image: {path.name}")
            for aabb in aabbs:
                x1 = aabb[0].item() / 416 * w
                y1 = aabb[1].item() / 416 * h
                x2 = aabb[2].item() / 416 * w
                y2 = aabb[3].item() / 416 * h
                cid = int(aabb[6].item())
                cpr = round(aabb[5].item(), 3)
                draw.text((x1, y1), str(cid), "#ffffff")
                draw.rectangle([(x1, y1), (x2, y2)], outline="#000000")
                print(f"{cid}: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}): {cpr}")
            img.save(results_dir.joinpath(path.name))


def main():
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        auto_scale_batch_size=False, accelerator="gpu", devices=1, logger=False
    )
    model = YoloV3Module(80)
    ds = SimpleDataset("testimgs")
    batch_size = 5
    dl = DataLoader(ds, batch_size, False)
    bsaabbs = trainer.predict(model, dl)
    show_results(ds, batch_size, bsaabbs)


if __name__ == "__main__":
    main()
