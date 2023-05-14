from argparse import ArgumentParser
from datetime import datetime

from lightning import Trainer
import torch
from torch.utils.data import DataLoader

import dataset_processing.coco_labels as coco_labels
from dataset_processing.datamodule import DataModule
from dataset_processing.simple_dataset import SimpleDataset
from model.yolov3 import YoloV3Module
from output_processing.display import output_results


OUTPUT_DIR = datetime.now().strftime("./detection_results/coco/%Y-%m-%d_%H-%M-%S")


def main(debug=False, show_num=0):
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
        output_results(results, 416, False, show_num, OUTPUT_DIR, coco_labels.labels)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-n", "--shownum", type=int, default=0)
    args = parser.parse_args()
    main(args.debug, args.shownum)
