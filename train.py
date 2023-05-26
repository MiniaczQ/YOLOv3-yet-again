from argparse import ArgumentParser, FileType
from datetime import datetime
from multiprocessing import cpu_count
from typing import IO, Optional, Union

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from dataset_processing.datamodule import DataModule
import model.metric_names as metric_names
from model.yolov3 import YoloV3Module


MODEL_CHECKPOINT_DIR = datetime.now().strftime("model_checkpoints/%Y-%m-%d_%H-%M-%S")
MODEL_CHECKPOINT_FILENAME = (
    "model-{epoch:02d}-{val_"
    + metric_names.avg_loss
    + ":.2f}-{val_"
    + metric_names.map_50_95
    + ":.2f}"
)


def main(
    loaded_checkpoint_path: Optional[Union[str, IO]] = None,
    dataset_root="./data/PKLot",
    max_epochs=500,
    debug=False,
    panet=False,
):
    if debug:
        torch.manual_seed(0)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        auto_scale_batch_size=True,
        accelerator="auto" if not debug else "cpu",
        devices=1,
        logger=True,
        max_epochs=max_epochs if not debug else 1,
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

    anchors = torch.tensor(
        [
            [[12, 14], [13, 17], [15, 15]],
            [[17, 19], [23, 20], [27, 25]],
            [[30, 15], [38, 21], [48, 36]],
        ]
    )
    model = (
        YoloV3Module(2, anchors=anchors, panet=panet)
        if loaded_checkpoint_path is None
        else YoloV3Module.load_from_checkpoint(loaded_checkpoint_path)
    )

    trainer.fit(
        model=model,
        datamodule=DataModule(dataset_root, cpu_count() if not debug else 0),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("loaded_checkpoint", nargs="?", type=FileType("rb"))
    parser.add_argument("--dataset", type=str, default="./data/PKLot")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--panet", action="store_true")
    args = parser.parse_args()
    main(args.loaded_checkpoint, args.dataset, args.epochs, args.debug, args.panet)
