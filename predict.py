from argparse import ArgumentParser, FileType
from datetime import datetime
from multiprocessing import cpu_count
from typing import IO, Union

from lightning import Trainer
import torch

from dataset_processing.datamodule import DataModule
from model.yolov3 import YoloV3Module
from output_processing.display import output_results


OUTPUT_DIR = datetime.now().strftime("./detection_results/pklot/%Y-%m-%d_%H-%M-%S")


def main(loaded_checkpoint_path: Union[str, IO], debug=False, show_num=0):
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
        output_results(results, 416, False, show_num, OUTPUT_DIR, ["empty", "occupied"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("loaded_checkpoint", type=FileType("rb"))
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-n", "--shownum", type=int, default=0)
    args = parser.parse_args()
    main(args.loaded_checkpoint, args.debug, args.shownum)
