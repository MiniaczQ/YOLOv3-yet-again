from datetime import datetime
from typing import IO, Union
from lightning import Trainer
from datamodule import DataModule
from multiprocessing import cpu_count
from yolov3 import YoloV3Module
from display import output_results
import torch
from argparse import ArgumentParser, FileType


OUTPUT_DIR = datetime.now().strftime("./detection_results/pklot/%Y-%m-%d_%H-%M-%S")


def main(loaded_checkpoint_path: Union[str, IO], debug=False):
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto" if not debug else "cpu",
        devices=1,
        logger=False,
        limit_predict_batches=32 if not debug else 1,
    )

    model = YoloV3Module.load_from_checkpoint(loaded_checkpoint_path)
    results = trainer.predict(
        model,
        datamodule=DataModule(cpu_count() if not debug else 0),
    )

    with torch.no_grad():
        output_results(results, 416, False, 0, OUTPUT_DIR, ["empty", "occupied"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("loaded_checkpoint", type=FileType("rb"))
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args.loaded_checkpoint, args.debug)
