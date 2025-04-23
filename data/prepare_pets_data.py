# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Script to preprocess Oxford Pets dataset for bounding box prediction."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from skimage import io
from tqdm import tqdm

# Infer ground-truth label from cat or dog species given in filename
cat_breeds = [
    "abyssinian",
    "bengal",
    "birman",
    "bombay",
    "british_shorthair",
    "egyptian_mau",
    "maine_coon",
    "persian",
    "ragdoll",
    "russian_blue",
    "siamese",
    "sphynx",
]

dog_breeds = [
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
]


def xml_to_csv(pths: list, img_dir: Path) -> pd.DataFrame:
    """Extracts the filenames and the bboxes from the xml_list."""
    print("[INFO] Gathering the filenames and bboxes")
    xml_list = []
    for xml_file in tqdm(pths):
        # Read in the xml file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for m in root.findall("object"):
            try:
                cname = root.find("filename").text.rpartition("_")[0]
                if cname in dog_breeds:
                    cidx = 1
                elif cname in cat_breeds:
                    cidx = 0
                else:
                    raise ValueError("Invalid image path.")
                value = (
                    # Extract the path to the image
                    str(img_dir / root.find("filename").text),
                    # Extract the bounding boxes
                    # 1. xmin
                    float(m[4][0].text),
                    # 2. ymin
                    float(m[4][1].text),
                    # 3. xmax
                    float(m[4][2].text),
                    # 4. ymax
                    float(m[4][3].text),
                    # class cat vs dogs
                    int(cidx),
                )
                # Try to read the image using skimage to ensure we can open it
                io.imread(str(img_dir / root.find("filename").text))
                xml_list.append(value)
            except Exception:
                pass
    col_n = ["filename", "xmin", "ymin", "xmax", "ymax", "class"]
    return pd.DataFrame(xml_list, columns=col_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        help="Path to the Oxford Pets dataset. " "Should contain images and annotations folders.",
    )
    config = parser.parse_args().__dict__

    # Get path to images
    root_path = config["path"]
    img_dir = Path(os.path.join(root_path, "images"))
    ims = list(img_dir.iterdir())
    ims = [str(pth) for pth in ims]
    img_pths = []
    print("[INFO] Gathering images")
    # Accept only the image files
    for im in tqdm(ims):
        if im.split(os.path.sep)[-1].split(".")[-1] == "jpg":
            img_pths.append(im)
    # Get annotations.
    annot_dir = Path(os.path.join(root_path, "annotations/xmls"))
    annots = list(annot_dir.iterdir())
    annots = [str(a) for a in annots]
    ann_pths = []
    print("[INFO] Creating the paths to the annotations ...")
    for a in tqdm(annots):
        for i in img_pths:
            # Check if the annotation file for an image is in
            # our verified img_pths or not
            i_pth = i.split(os.path.sep)[-1].split(".")[0]
            a_pth = a.split(os.path.sep)[-1].split(".")[0]
            if i_pth == a_pth:
                ann_pths.append(a)
    print("Annotation files found : ", len(ann_pths))
    # Process annotations.
    df = xml_to_csv(ann_pths, img_dir)
    df.to_csv(os.path.join("data", "pet_bb_annotations.csv"), index=False)
