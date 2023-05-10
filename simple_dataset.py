from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from torchvision import transforms

from processing import ResizeKeepRatio, PadToSquare


class SimpleDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.images = list(self.root.rglob("*.jpg"))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ResizeKeepRatio(416),
                PadToSquare(),
            ]
        )

    def __getitem__(self, i):
        img_path = self.images[i]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

    def get_raw(self, i):
        img_path = self.images[i]
        img = Image.open(img_path).convert("RGB")
        return img_path, img

    def __len__(self):
        return len(self.images)
