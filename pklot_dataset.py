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
    def __init__(self, root, img_transform=None, ann_transform=None):
        self.root = Path(root)
        self.image_paths = list(self.root.rglob("*.jpg"))
        self.img_transform = img_transform
        self.ann_transform = ann_transform

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = Image.open(image_path).convert("RGB")
        if self.img_transform is not None:
            image = self.img_transform(image)

        annotations_path = image_path.with_suffix(".xml")
        annotations_xml = ET.parse(annotations_path)
        annotations = list(_iter_spaces(annotations_xml.iterfind("space")))
        if self.ann_transform is not None:
            annotations = self.ann_transform(annotations)

        return image, annotations

    def __len__(self):
        return len(self.image_paths)

    # def __repr__(self) -> str:
    #    format_string = f"{self.__class__.__name__}(root={self.root}"
    #    if self.img_transform is not None:
    #        format_string += f"img_transform={self.img_transform.__repr__()}"
    #    if self.ann_transform is not None:
    #        format_string += f"ann_transform={self.ann_transform.__repr__()}"
    #    format_string += ")"
    #    return format_string


def main():
    ds = PkLotDataset("data/pklot")
    print(len(ds))
    print(ds[0])


if __name__ == "__main__":
    main()
