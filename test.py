from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from datamodule import Datamodule
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
        num_sanity_val_steps=0,
        benchmark=False,
    )
    model = YoloV3Module(2)
    # model.load_from_checkpoint(
    #     "model_checkpoints/2023-05-13_04-55-47/last/model-epoch=04-val_loss_mean=0.27-val_map_50_95=0.00.ckpt"
    # )
    dm = Datamodule(0, size_limit=1)  # TODO: use number of cores
    dm.batch_size = 4
    dm.prepare_data()
    dm.setup()
    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
