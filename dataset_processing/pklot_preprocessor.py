from argparse import ArgumentParser
from os import remove
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# Preprocess PKLot labels
# Skip invalid box definitions, not all visible parking spaces are labeled anyways
# Labels are saved as .data files with tensor([[occupied, min_x, min_y, max_x, max_y]]) format
# TODO run automatically if needed
def preprocess(root, silent=True, remove_old=False):
    if remove_old:
        for file in Path(root).rglob("*.data"):
            remove(file)
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
                try:
                    occupied = int(space.attrib["occupied"])
                except:
                    print(f"\tSkipping bbox without occupied attribute for image {image}")
                    continue
                any_point_found = False
                min_x, min_y = IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1
                max_x, max_y = 0, 0
                if (contour := space.find("contour")) is not None:
                    for point in contour.iterfind("point"):
                        any_point_found = True
                        x = int(point.attrib["x"])
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        y = int(point.attrib["y"])
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
                if any_point_found:
                    data.append([occupied, min_x, min_y, max_x, max_y])
        except Exception as e:
            if not silent:
                print(f"Image {image} encountered exception {type(e).__name__}: {e}")
            errors += 1
            continue
        if len(data) < 1:
            print(f"Skipping image {image} because no labels were correct")
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root", type=str, nargs="?", default="./data/PKLot")
    parser.add_argument("-s", "--silent", action="store_true")
    parser.add_argument("-r", "--removeold", action="store_true")
    args = parser.parse_args()
    preprocess(args.root, args.silent, args.removeold)
