import lightning as pl
import numpy as np
import torch
from torch import Generator
from torch.utils.data import random_split, DataLoader, ConcatDataset, Dataset
from torchvision import transforms

from pklot_dataset import PkLotDataset
from processing import PadToSquare, ResizeKeepRatio, NormalizeBbox


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers=2,
        *,
        size_limit=0,
        train_val_seed=2136,
        test_seed=2136,
        size_limit_seed=2136,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.size_limit = size_limit
        self.train_val_seed = train_val_seed
        self.test_seed = test_seed
        self.size_limit_seed = size_limit_seed
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

    def setup(
        self,
        stage=None,
    ):
        pklot_dataset = PkLotDataset(
            "data/pklot", self.img_transform, self.ann_transform
        )

        dataset = ConcatDataset([pklot_dataset])
        # Split 8:1:1
        dataset, self.test_dataset = random_split(
            dataset, [9 / 10, 1 / 10], Generator().manual_seed(self.test_seed)
        )
        self.train_dataset, self.val_dataset = random_split(
            dataset, [8 / 9, 1 / 9], Generator().manual_seed(self.train_val_seed)
        )

        if self.size_limit > 0:

            def _split_size_limit(dataset: Dataset):
                sample_count = self.size_limit * self.batch_size

                return random_split(
                    dataset,
                    [
                        min(sample_count, len(dataset)),
                        max(len(dataset) - sample_count, 0),
                    ],
                    Generator().manual_seed(self.size_limit_seed),
                )[0]

            self.test_dataset = _split_size_limit(self.test_dataset)
            self.train_dataset = _split_size_limit(self.train_dataset)
            self.val_dataset = _split_size_limit(self.val_dataset)

    # Custom collate function for combining multiple images & labels into a batch
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

    # TODO: prepare a separate set for prediction
    def predict_dataloader(self):
        return self.test_dataloader()
