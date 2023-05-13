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
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss_mean",
                dirpath=datetime.now().strftime(
                    "model_checkpoints/%Y-%m-%d_%H-%M-%S/loss"
                ),
                filename="model-{epoch:02d}-{val_loss_mean:.2f}-{val_map_mean:.2f}",
                save_top_k=3,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_map_mean",
                dirpath=datetime.now().strftime(
                    "model_checkpoints/%Y-%m-%d_%H-%M-%S/map"
                ),
                filename="model-{epoch:02d}-{val_loss_mean:.2f}-{val_map_mean:.2f}",
                save_top_k=3,
                mode="min",
            ),
            ModelCheckpoint(
                dirpath=datetime.now().strftime(
                    "model_checkpoints/%Y-%m-%d_%H-%M-%S/last"
                ),
                filename="model-{epoch:02d}-{val_loss_mean:.2f}-{val_map_mean:.2f}",
                save_top_k=1,
            ),
        ],
        # overfit_batches=1,
        benchmark=False,
    )
    model = YoloV3Module(2)
    # model.load_from_checkpoint(
    #     "model_checkpoints/2023-05-12_08-00-41/model-epoch=01-val_loss_mean=5.16.ckpt"
    # )
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # ds = SimpleDataset("testimgs")
    # batch_size = 5
    # dl = DataLoader(ds, batch_size, False)
    # bsaabbs = trainer.predict(model, dl)
    # show_results(ds, batch_size, bsaabbs)
    # summary(model.model, (1, 3, 416, 416))
    dm = Datamodule(0)
    dm.batch_size = 32
    dm.prepare_data()
    dm.setup()
    # dl = dm.train_dataloader(0)
    # data = next(iter(dl))
    # print(data)
    # model.training_step(data, 0)
    # tuner = trainer.tuner
    # tuner.scale_batch_size(model=model, datamodule=dm)
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
