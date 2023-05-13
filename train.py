from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from datamodule import DataModule
from yolov3 import YoloV3Module
from lightning.pytorch.callbacks import ModelCheckpoint
from datetime import datetime
from multiprocessing import cpu_count

debug = False
model_checkpoint_dir = datetime.now().strftime(
    "model_checkpoints/%Y-%m-%d_%H-%M-%S/loss"
)
model_checkpoint_filename = "model-{epoch:02d}-{val_loss_mean:.2f}-{val_map_50_95:.2f}"


def main():
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        auto_scale_batch_size=True,
        accelerator="auto",
        devices=1,
        logger=True,
        max_epochs=500,
        num_sanity_val_steps=0,
        benchmark=False,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss_mean",
                dirpath=model_checkpoint_dir,
                filename=model_checkpoint_filename,
                save_top_k=3,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_map_50_95",
                dirpath=model_checkpoint_dir,
                filename=model_checkpoint_filename,
                save_top_k=3,
                mode="max",
            ),
            ModelCheckpoint(
                dirpath="model_checkpoints/last",
                filename="last",
                save_top_k=1,
            ),
        ],
    )

    model = YoloV3Module(2)

    trainer.fit(
        model=model,
        datamodule=DataModule(cpu_count() if not debug else 0),
    )


if __name__ == "__main__":
    main()
