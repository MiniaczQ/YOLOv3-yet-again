from argparse import ArgumentParser, FileType
from datetime import datetime
from multiprocessing import cpu_count
from typing import IO, Optional, Union

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from datamodule import DataModule
import metric_names
from yolov3 import YoloV3Module


MODEL_CHECKPOINT_DIR = datetime.now().strftime("model_checkpoints/%Y-%m-%d_%H-%M-%S")
MODEL_CHECKPOINT_FILENAME = (
    "model-{epoch:02d}-{val_"
    + metric_names.avg_loss
    + ":.2f}-{val_"
    + metric_names.map_50_95
    + ":.2f}"
)


def main(loaded_checkpoint_path: Optional[Union[str, IO]] = None, debug=False):
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        auto_scale_batch_size=True,
        accelerator="auto" if not debug else "cpu",
        devices=1,
        logger=True,
        max_epochs=500 if not debug else 1,
        limit_train_batches=16 * 9 if not debug else 1,
        limit_val_batches=16 if not debug else 1,
        num_sanity_val_steps=0,
        benchmark=False,
        callbacks=[
            ModelCheckpoint(
                monitor="val_" + metric_names.avg_loss,
                dirpath=MODEL_CHECKPOINT_DIR + "/loss",
                filename=MODEL_CHECKPOINT_FILENAME,
                save_top_k=3,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_" + metric_names.map_50_95,
                dirpath=MODEL_CHECKPOINT_DIR + "/map",
                filename=MODEL_CHECKPOINT_FILENAME,
                save_top_k=3,
                mode="max",
            ),
            ModelCheckpoint(
                dirpath=MODEL_CHECKPOINT_DIR + "/last",
                filename="last",
                save_top_k=1,
            ),
        ]
        if not debug
        else None,
    )

    model = (
        YoloV3Module(2)
        if loaded_checkpoint_path is None
        else YoloV3Module.load_from_checkpoint(loaded_checkpoint_path)
    )

    trainer.fit(
        model=model,
        datamodule=DataModule(cpu_count() if not debug else 0),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("loaded_checkpoint", nargs="?", type=FileType("rb"))
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args.loaded_checkpoint, args.debug)
