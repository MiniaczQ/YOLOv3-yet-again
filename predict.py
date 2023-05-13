from datetime import datetime
from lightning import Trainer
from datamodule import DataModule
from multiprocessing import cpu_count
from yolov3 import YoloV3Module
from display import output_results
import torch


debug = False
checkpoint_filename = "model_checkpoints/prepared.ckpt"
output_dir = datetime.now().strftime("./detection_results/pklot/%Y-%m-%d_%H-%M-%S")


def main():
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        limit_predict_batches=32,
    )

    model = YoloV3Module.load_from_checkpoint(checkpoint_filename)
    results = trainer.predict(
        model,
        datamodule=DataModule(cpu_count() if not debug else 0),
    )

    with torch.no_grad():
        output_results(results, 416, False, 0, output_dir, ["empty", "occupied"])


if __name__ == "__main__":
    main()
