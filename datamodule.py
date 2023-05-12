import lightning as pl
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torchvision import transforms
from torch import Generator
from pklot_dataset import PkLotDataset

from processing import PadToSquare, ResizeKeepRatio, NormalizeBbox
import pklot_preprocessor


class Datamodule(pl.LightningDataModule):
    def __init__(self, num_workers=2):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = 16
        self.unscaled_size = (1280, 720)
        self.img_size = (416, 416)
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ResizeKeepRatio(416),
                PadToSquare(114),
            ]
        )
        self.ann_transform = transforms.Compose([NormalizeBbox(self.unscaled_size)])

    def prepare_data(self):
        return
        pklot_preprocessor.preprocess("data/pklot")

    def setup(self, stage=None, train_val_seed=2136, test_seed=2136):
        pklot_dataset = PkLotDataset(
            "data/pklot", self.img_transform, self.ann_transform
        )
        dataset = ConcatDataset([pklot_dataset])

        dataset, self.test_dataset = random_split(
            dataset, [9 / 10, 1 / 10], Generator().manual_seed(test_seed)
        )

        self.train_dataset, self.val_dataset = random_split(
            dataset, [8 / 9, 1 / 9], Generator().manual_seed(train_val_seed)
        )

    @staticmethod
    def _collate_fn(batch):
        image_batch = torch.stack([elem[0] for elem in batch], 0)
        annotation_batch = torch.cat(
            [
                torch.cat(
                    (
                        torch.ones(annotations.size(0), 1) * batch_index,
                        annotations,
                    ),
                    1,
                )
                for batch_index, annotations in enumerate([elem[1] for elem in batch])
            ],
            0,
        )
        path_batch = [elem[2] for elem in batch]
        raw_image_batch = [elem[3] for elem in batch]
        return (image_batch, annotation_batch, path_batch, raw_image_batch)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
