from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .processing import ResizeKeepRatio, PadToSquare


# Simple dataset for testing
class SimpleDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.images = list(self.root.rglob("*.jpg"))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ResizeKeepRatio(416),
                PadToSquare(0.447),
            ]
        )

    def __getitem__(self, i):
        img_path = self.images[i]
        raw_img = Image.open(img_path).convert("RGB")
        img = self.transform(raw_img)
        annotations = torch.tensor([])
        return img, annotations, str(img_path.name), raw_img

    def __len__(self):
        return len(self.images)
