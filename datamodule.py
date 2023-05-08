import lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torch import manual_seed


class Datamodule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.img_size = (416, 416)

    def prepare_data(self):
        pass  # TODO maybe

    def setup(self, stage=None, seed=2136):
        manual_seed(seed)
        dataset = ImageNetDogsDataLoader(
            "ImageNetDogs/Images", "ImageNetDogs/train_list.mat", self.get_transform()
        )
        self.dataset_train, self.dataset_val = random_split(dataset, [0.9, 0.1])
        self.dataset_test = ImageNetDogsDataLoader(
            "ImageNetDogs/Images", "ImageNetDogs/test_list.mat", self.get_transform()
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [
                        0.485,
                        0.456,
                        0.406,
                    ],  # Values from https://d2l.ai/chapter_computer-vision/kaggle-dog.html
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
