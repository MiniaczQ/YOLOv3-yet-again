from datetime import datetime
from pathlib import Path
from lightning import Trainer
import torch
from torch.utils.data import DataLoader
from yolov3 import YoloV3Module
from display import display_dir, process_results
from simple_dataset import SimpleDataset
import coco_labels


def main():
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        auto_scale_batch_size=False, accelerator="auto", devices=1, logger=False
    )
    model = YoloV3Module(80)  # Auto-load COCOv3
    model.conf_threshold = 0.3
    model.iou_threshold = 0.5
    ds = SimpleDataset("data/COCO")
    batch_size = 32
    dl = DataLoader(
        ds,
        batch_size,
        False,
        collate_fn=lambda x: (
            list([i[0] for i in x]),
            torch.stack([i[1] for i in x]),
            list([i[2] for i in x]),
        ),
    )
    results = trainer.predict(model, dl)

    out_dir = Path(datetime.now().strftime("detection_results/coco/%Y_%m_%d_%H_%M_%S"))
    with torch.no_grad():
        process_results(results, 416, True, False, out_dir, coco_labels.labels)
    display_dir(out_dir)


if __name__ == "__main__":
    main()
