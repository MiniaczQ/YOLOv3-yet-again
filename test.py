from multiprocessing import cpu_count
from lightning import Trainer
from datamodule import DataModule
from yolov3 import YoloV3Module
import torch


debug = False
checkpoint_filename = "model_checkpoints/prepared.ckpt"


def main():
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto" if not debug else "cpu",
        devices=1,
        logger=True,
        benchmark=False,
        limit_test_batches=16 if not debug else 1,
    )

    model = YoloV3Module.load_from_checkpoint(checkpoint_filename)
    trainer.test(
        model=model,
        datamodule=DataModule(cpu_count() if not debug else 0),
    )


if __name__ == "__main__":
    main()
