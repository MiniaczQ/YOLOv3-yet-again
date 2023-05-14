from datetime import datetime
from lightning import Trainer
import torch
from torch.utils.data import DataLoader
from datamodule import DataModule
from yolov3 import YoloV3Module
from display import output_results
from simple_dataset import SimpleDataset
import coco_labels

debug = False
output_dir = datetime.now().strftime("./detection_results/coco/%Y-%m-%d_%H-%M-%S")


def main():
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto" if not debug else "cpu", devices=1, logger=False
    )

    model = YoloV3Module(80)  # Auto-load COCOv3
    model.conf_threshold = 0.3
    model.iou_threshold = 0.5

    ds = SimpleDataset("data/COCO")
    dl = DataLoader(
        ds, 32 if not debug else 1, False, collate_fn=DataModule._collate_fn
    )

    results = trainer.predict(model, dl)

    with torch.no_grad():
        output_results(results, 416, False, 0, output_dir, coco_labels.labels)


if __name__ == "__main__":
    main()
