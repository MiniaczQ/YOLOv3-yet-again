from lightning import Trainer
import torch
from datamodule import Datamodule
from yolov3 import YoloV3Module
from display import process_results


def main():
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        auto_scale_batch_size=False, accelerator="auto", devices=1, logger=False
    )
    model = YoloV3Module(2)
    model.load_from_checkpoint(
        "model_checkpoints/2023-05-13_00-02-31/model-epoch=01-val_loss_mean=0.42.ckpt"
    )
    model.conf_threshold = 0.3
    model.iou_threshold = 0.5
    dm = Datamodule(0)
    dm.batch_size = 32
    dm.prepare_data()
    dm.setup()
    results = trainer.predict(model, dm.test_dataloader(1))

    with torch.no_grad():
        process_results(
            results, 416, False, 0, "./detection_results/pklot/", ["empty", "occupied"]
        )


if __name__ == "__main__":
    main()
