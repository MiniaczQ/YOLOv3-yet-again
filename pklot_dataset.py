from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset


# Load PKLot with labels
class PkLotDataset(Dataset):
    def __init__(self, root, img_transform=None, ann_transform=None):
        self.root = Path(root)
        self.ann_paths = list(self.root.rglob("*.data"))
        self.img_transform = img_transform
        self.ann_transform = ann_transform

    def __getitem__(self, i):
        data_paths = self.ann_paths[i]
        img_path = data_paths.with_suffix(".jpg")
        img = raw_img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)

        ann = torch.load(data_paths)
        if self.ann_transform is not None:
            ann = self.ann_transform(ann)

        return img, ann, str(img_path.name), raw_img

    def __len__(self):
        return len(self.ann_paths)


def main():
    ds = PkLotDataset("data/pklot")
    print(len(ds))
    for i in range(len(ds)):
        ds[i]


if __name__ == "__main__":
    main()
