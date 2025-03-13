#! /usr/bin/env python3

""" tile_split_ready.py

    This script is used to process the CVAT annotated datasets, tile the images and labels, and split the dataset into train/val/test.
"""

import os
import sys
import yaml
import glob
import torch
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.image_tiler import ImageTiler
from utils.file_splitter import DatasetSplitter

# Directories
SSD_PATH = "/media/java/RRAP03"

# Tiling parameters
TILE_SIZE = (640, 640)
OVERLAP = 50 # % overlap
CLASSES = None # None for all classes. Can select classes => [0, 1, 2, 3]

# Splitting ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

base_data_dir = os.path.join(SSD_PATH, "exported_from_cvat")
outputs_dir = os.path.join(SSD_PATH, "outputs/train")

tiled_output_dir = os.path.join(outputs_dir, "tiled_dataset")
split_output_dir = os.path.join(outputs_dir, "split_dataset")
yaml_output_path = os.path.join(outputs_dir, "cgras_data.yaml")

# Loop through multiple dataset folders
dataset_folders = sorted(glob.glob(os.path.join(base_data_dir, "*")))  # Assuming each dataset is in a separate folder

for dataset_path in dataset_folders:
    dataset_name = os.path.basename(dataset_path)
    print(f"\nProcessing dataset: {dataset_name}")

    # Step 1: Tile Images and Labels
    tiled_dataset_path = os.path.join(tiled_output_dir, dataset_name)
    os.makedirs(tiled_dataset_path, exist_ok=True)

    tiler = ImageTiler(
        ssd_path=SSD_PATH,
        data_path=dataset_path,
        output_path=tiled_dataset_path,
        tile_size=TILE_SIZE,
        overlap_percent=OVERLAP,
        max_files=16382,
        wanted_classes=CLASSES
    )
    tiler.tile_images()
    print(f"Tiling completed for {dataset_name}")

    # Step 2: Split Dataset into Train/Val/Test
    split_dataset_path = os.path.join(split_output_dir, dataset_name)
    os.makedirs(split_dataset_path, exist_ok=True)

    splitter = DatasetSplitter(
        ssd_path=SSD_PATH,
        data_location=tiled_dataset_path,
        save_dir=split_dataset_path,
        train_ratio=TRAIN_SPLIT,
        valid_ratio=VAL_SPLIT,
        test_ratio=TEST_SPLIT
    )
    splitter.split_dataset()
    print(f"Splitting completed for {dataset_name}")

# Step 3: Update cgras_data.yaml
dataset_dirs = sorted(glob.glob(os.path.join(split_output_dir, "*")))
final_yaml_data = {
    "path": split_output_dir,
    "train": [f"{os.path.basename(dir)}/train/images" for dir in dataset_dirs],
    "val": [f"{os.path.basename(dir)}/valid/images" for dir in dataset_dirs],
    "test": [f"{os.path.basename(dir)}/test/images" for dir in dataset_dirs],
    "names": {0: "alive", 1: "dead", 2: "mask_live", 3: "mask_dead"}
}

with open(yaml_output_path, "w") as yaml_file:
    yaml.dump(final_yaml_data, yaml_file, default_flow_style=False)

print(f"\n Dataset processing complete. The dataset details have been saved to {yaml_output_path}.")

# Step 4: Confirm Training Parameters with User
print("\n Training Configuration:")
print(f"   - Dataset path: {split_output_dir}")
print(f"   - Train images: {final_yaml_data['train']}")
print(f"   - Validation images: {final_yaml_data['val']}")
print(f"   - Test images: {final_yaml_data['test']}")