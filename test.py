from multiprocessing import cpu_count
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from datamodule import DataModule
from yolov3 import YoloV3Module
from datetime import datetime
import torch


debug = False
checkpoint_filename = "model_checkpoints/prepared.ckpt"


def main():
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=True,
        benchmark=False,
        limit_test_batches=16,
    )

    model = YoloV3Module.load_from_checkpoint(checkpoint_filename)

    metrics = trainer.test(
        model=model,
        datamodule=DataModule(cpu_count() if not debug else 0),
    )
    print(f"mAP@0.5:0.95: {metrics[0]['test_map_50_95']}")
    print(f"mAP@0.5: {metrics[0]['test_map_50']}")
    print(f"mAP@0.75: {metrics[0]['test_map_75']}")


if __name__ == "__main__":
    main()
