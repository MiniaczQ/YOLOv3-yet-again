from argparse import ArgumentParser, FileType
from multiprocessing import cpu_count
from typing import IO, Union

from lightning import Trainer
import torch

from dataset_processing.datamodule import DataModule
from model.yolov3 import YoloV3Module


def main(
    loaded_checkpoint_path: Union[str, IO],
    dataset_root="./data/PKLot",
    sample_count=16,
    debug=False,
):
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto" if not debug else "cpu",
        devices=1,
        logger=True,
        benchmark=False,
        limit_test_batches=sample_count if not debug else 1,
    )

    model = YoloV3Module.load_from_checkpoint(loaded_checkpoint_path)
    trainer.test(
        model=model,
        datamodule=DataModule(dataset_root, cpu_count() if not debug else 0),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("loaded_checkpoint", type=FileType("rb"))
    parser.add_argument("--dataset", type=str, default="./data/PKLot")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args.loaded_checkpoint, args.dataset, args.samples, args.debug)
