from lightning import Trainer
import torch
from torch.utils.data import DataLoader

from yolov3 import YoloV3Module
from display import show_results
from simple_dataset import SimpleDataset


def main():
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        auto_scale_batch_size=False, accelerator="gpu", devices=1, logger=False
    )
    model = YoloV3Module(80)
    ds = SimpleDataset("test_images")
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
    with torch.no_grad():
        show_results(results, 416)


if __name__ == "__main__":
    main()
