from multiprocessing import cpu_count
from typing import IO, Union
from lightning import Trainer
from datamodule import DataModule
from yolov3 import YoloV3Module
import torch
from argparse import ArgumentParser, FileType


def main(loaded_checkpoint_path: Union[str, IO], debug=False):
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto" if not debug else "cpu",
        devices=1,
        logger=True,
        benchmark=False,
        limit_test_batches=16 if not debug else 1,
    )

    model = YoloV3Module.load_from_checkpoint(loaded_checkpoint_path)
    trainer.test(
        model=model,
        datamodule=DataModule(cpu_count() if not debug else 0),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("loaded_checkpoint", type=FileType("rb"))
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args.loaded_checkpoint, args.debug)
