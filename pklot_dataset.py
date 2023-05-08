from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET


def _iter_points(points):
    for point in points:
        x = point.attrib["x"]
        y = point.attrib["y"]
        yield (x, y)


def _iter_spaces(spaces):
    for space in spaces:
        occupied = space.attrib["occupied"]
        points = list(_iter_points(space.find("contour").iterfind("point")))
        yield (occupied, points)


class PkLotDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.image_paths = list(self.root.rglob("*.jpg"))

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        annotations_path = image_path.with_suffix(".xml")
        annotations_xml = ET.parse(annotations_path)
        annotations = list(_iter_spaces(annotations_xml.iterfind("space")))
        image = Image.open(image_path).convert("RGB")
        return image, annotations

    def __len__(self):
        return len(self.image_paths)


def main():
    ds = PkLotDataset("data/pklot")
    print(len(ds))
    print(ds[0])


if __name__ == "__main__":
    main()
