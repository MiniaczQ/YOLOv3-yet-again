from lightning import Trainer

from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from datamodule import DataModule

from yolov3 import YoloV3Module

from lightning.pytorch.callbacks import ModelCheckpoint

from datetime import datetime


def main():
    torch.manual_seed(123)

    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        auto_scale_batch_size=False,
        accelerator="gpu",
        devices=1,
        logger=True,
        max_epochs=500,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss_mean",
                dirpath=datetime.now().strftime(
                    "model_checkpoints/%Y-%m-%d_%H-%M-%S/loss"
                ),
                filename="model-{epoch:02d}-{val_loss_mean:.2f}-{val_map_50_95:.2f}",
                save_top_k=3,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_map_50_95",
                dirpath=datetime.now().strftime(
                    "model_checkpoints/%Y-%m-%d_%H-%M-%S/map"
                ),
                filename="model-{epoch:02d}-{val_loss_mean:.2f}-{val_map_50_95:.2f}",
                save_top_k=3,
                mode="max",
            ),
            ModelCheckpoint(
                dirpath=datetime.now().strftime(
                    "model_checkpoints/%Y-%m-%d_%H-%M-%S/last"
                ),
                filename="model-{epoch:02d}-{val_loss_mean:.2f}-{val_map_50_95:.2f}",
                save_top_k=1,
            ),
        ],
        num_sanity_val_steps=0,
        # overfit_batches=1,
        benchmark=False,
    )

    model = YoloV3Module(2)

    dm = DataModule(12)
    dm.batch_size = 16
    # dm.prepare_data()
    dm.setup()

    # tuner = trainer.tuner
    # tuner.scale_batch_size(model=model, datamodule=dm)
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
