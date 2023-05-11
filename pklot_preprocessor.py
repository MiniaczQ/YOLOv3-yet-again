from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import numpy as np
from os import remove


def preprocess(root, silent=True):
    printed = True
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
                min_x, min_y = 2**32 - 1, 2**32 - 1
                max_x, max_y = 0, 0
                for point in space.find("contour").iterfind("point"):
                    x = int(point.attrib["x"])
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    y = int(point.attrib["y"])
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                data.append([occupied, min_x, min_y, max_x, max_y])
        except Exception as e:
            if not silent:
                print(f"Image {image} encountered exception {type(e).__name__}: {e}")
            errors += 1
            continue
        data = torch.tensor(data)
        if not printed:
            printed = True
            print(data)
        data_file = image.with_suffix(".data")
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
