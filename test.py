from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from datamodule import DataModule
from yolov3 import YoloV3Module
from datetime import datetime
import torch


def main():
    torch.manual_seed(123)

    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        auto_scale_batch_size=False,
        accelerator="gpu",
        devices=1,
        logger=True,
        max_epochs=500,
        num_sanity_val_steps=0,
        benchmark=False,
    )

    model = YoloV3Module(2)
    model = model.load_from_checkpoint("model_checkpoints/prepared.ckpt")

    dm = DataModule(12)
    dm.batch_size = 16
    # dm.prepare_data()
    dm.setup()

    metrics = trainer.test(model=model, datamodule=dm)
    print(f"mAP@0.5:0.95: {metrics['val_map_50_95']}")
    print(f"mAP@0.5: {metrics['val_map_50']}")


if __name__ == "__main__":
    main()
