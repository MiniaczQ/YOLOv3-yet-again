import lightning as pl
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torchvision import transforms
from torch import Generator
from pklot_dataset import PkLotDataset

from processing import pad, resize


class Datamodule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.img_size = (416, 416)

    def prepare_data(self):
        pass  # TODO maybe

    def setup(self, stage=None, train_val_seed=2136, test_seed=2136):
        pklot_dataset = PkLotDataset("data/pklot", self.get_transform())
        dataset = ConcatDataset([pklot_dataset])

        dataset, self.test_dataset = random_split(
            dataset, [9 / 10, 1 / 10], Generator.manual_seed(test_seed)
        )

        self.train_dataset, self.val_dataset = random_split(
            dataset, [8 / 9, 1 / 9], Generator.manual_seed(train_val_seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Lambda(pad),
                transforms.Lambda(resize),
                transforms.Lambda(lambda t: t.cuda().unsqueeze(0)),
            ]
        )
