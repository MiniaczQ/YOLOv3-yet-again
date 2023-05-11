from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from datamodule import Datamodule

from yolov3 import YoloV3Module
from lightning.pytorch.callbacks import ModelCheckpoint

MODEL_CKPT_PATH = "model/"
MODEL_CKPT = "model-{epoch:02d}-{val_loss:.2f}"

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=MODEL_CKPT_PATH,
    filename=MODEL_CKPT,
    save_top_k=3,
    mode="min",
)


def main():
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
                dirpath=datetime.now().strftime("model/%Y-%m-%d_%H-%M-%S/"),
                filename="model-{epoch:02d}-{val_loss_mean:.2f}",
                save_top_k=3,
                mode="min",
            )
        ],
        # overfit_batches=1,
        benchmark=False,
    )
    full_model = YoloV3Module(80)
    model = YoloV3Module(2)
    model.model.backbone = full_model.model.backbone
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # ds = SimpleDataset("testimgs")
    # batch_size = 5
    # dl = DataLoader(ds, batch_size, False)
    # bsaabbs = trainer.predict(model, dl)
    # show_results(ds, batch_size, bsaabbs)
    # summary(model.model, (1, 3, 416, 416))
    dm = Datamodule(8)
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
