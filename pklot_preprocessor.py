from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import numpy as np
from os import remove


def preprocess(root, silent=True):
    root = Path(root)
    images = list(root.rglob("*.jpg"))
    total = 0
    successful = 0
    errors = 0
    for image in images:
        total += 1
        data = []
        try:
            annotations_path = image.with_suffix(".xml")
            annotations_xml = ET.parse(annotations_path)
            for space in annotations_xml.iterfind("space"):
                occupied = int(space.attrib["occupied"])
                data.append(occupied)
                for point in space.find("contour").iterfind("point"):
                    x = int(point.attrib["x"])
                    data.append(x)
                    y = int(point.attrib["y"])
                    data.append(y)
        except:
            if not silent:
                print(f"Image {image} encountered exception")
            errors += 1
            continue
        data = torch.tensor(data)
        data_file = image.with_suffix(".data")
        remove(data_file)
        torch.save(data, data_file)
        successful += 1
    if not silent:
        print(f"Total: {total}")
        print(f"Successful: {successful}")
        print(f"Errors: {errors}")


def main():
    preprocess("data/pklot", False)


if __name__ == "__main__":
    main()
