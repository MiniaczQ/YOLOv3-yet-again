from lightning import Trainer
import torch
from datamodule import Datamodule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from simple_dataset import SimpleDataset

from yolov3 import YoloV3Module


def main():
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        auto_scale_batch_size=False, accelerator="gpu", devices=1, logger=False
    )
    model = YoloV3Module(80)
    # dm = Datamodule()
    # dm.batch_size = 8
    # dm.prepare_data()
    # dm.setup()
    # dl = dm.test_dataloader(0)
    # imgs = [img[0] for img in next(iter(dl))]
    ds = SimpleDataset("testimgs")
    dl = DataLoader(ds, 6, False)
    bsaabbs = trainer.predict(model, dl)
    for baabbs in bsaabbs:
        for aabbs in baabbs:
            # print(aabbs)
            print()


if __name__ == "__main__":
    main()
