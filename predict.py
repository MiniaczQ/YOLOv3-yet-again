from lightning import Trainer
from datamodule import DataModule
from yolov3 import YoloV3Module
from display import process_results
import torch


def main():
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        auto_scale_batch_size=False, accelerator="auto", devices=1, logger=False
    )

    model = YoloV3Module(2)
    model = model.load_from_checkpoint("model_checkpoints/prepared.ckpt")

    dm = DataModule(12)
    dm.batch_size = 32
    # dm.prepare_data()
    dm.setup()

    results = trainer.predict(model, dm.test_dataloader(1))

    with torch.no_grad():
        process_results(
            results, 416, False, 0, "./detection_results/pklot/", ["empty", "occupied"]
        )


if __name__ == "__main__":
    main()
