from lightning import Trainer
import torch
from datamodule import Datamodule
from torch.utils.data import DataLoader, Dataset

from yolov3 import YoloV3Module


def main():
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        auto_scale_batch_size=False, accelerator="cpu", devices=1, logger=False
    )
    model = YoloV3Module()
    dm = Datamodule()
    dm.batch_size = 1
    dm.prepare_data()
    dm.setup()
    dl = dm.test_dataloader(1)
    img, labels = next(iter(dl))
    dl = DataLoader(img, 1, False)
    trainer.predict(model, dl)


if __name__ == "__main__":
    main()
