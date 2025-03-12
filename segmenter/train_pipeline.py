#! /usr/bin/env python3

""" train_pipeline.py

    This script is used to process the CVAT annotated datasets, tile the images and labels, split the dataset into train/val/test and train a model.
"""

import os
import sys
import yaml
import glob
import torch
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing.image_tiler import ImageTiler
from image_processing.file_splitter import DatasetSplitter


# Directories
base_data_dir = "/home/java/data/exported_from_cvat"  # Folder containing one or multiple datasets
outputs_dir = "/home/java/data/outputs/train_pipeline"
tiled_output_dir = os.path.join(outputs_dir, "tiled_dataset")
split_output_dir = os.path.join(outputs_dir, "split_dataset")
yaml_output_path = os.path.join(outputs_dir, "cgras_data.yaml")

# Tiling parameters
tile_size = (640, 640)
overlap_percent = 50
classes = None # None for all classes. Can select classes => [0, 1, 2, 3]

# Splitting ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Training parameters
# model_path = "yolov8n-seg.pt"
model_path = "/home/java/data/best.pt"
pretrained = True
epochs = 16
batch_size = 16
workers = 4
save_period = 2
patience = 3

# Loop through multiple dataset folders
dataset_folders = sorted(glob.glob(os.path.join(base_data_dir, "*")))  # Assuming each dataset is in a separate folder

for dataset_path in dataset_folders:
    dataset_name = os.path.basename(dataset_path)
    print(f"\nProcessing dataset: {dataset_name}")

    # Step 1: Tile Images and Labels
    tiled_dataset_path = os.path.join(tiled_output_dir, dataset_name)
    os.makedirs(tiled_dataset_path, exist_ok=True)

    tiler = ImageTiler(
        tile_size=tile_size,
        overlap_percent=overlap_percent,
        data_path=dataset_path,
        output_path=tiled_dataset_path,
        max_files=16382,
        wanted_classes=classes
    )
    tiler.tile_images()
    print(f"Tiling completed for {dataset_name}")

    # Step 2: Split Dataset into Train/Val/Test
    split_dataset_path = os.path.join(split_output_dir, dataset_name)
    os.makedirs(split_dataset_path, exist_ok=True)

    splitter = DatasetSplitter(
        data_location=tiled_dataset_path,
        save_dir=split_dataset_path,
        train_ratio=train_ratio,
        valid_ratio=val_ratio,
        test_ratio=test_ratio
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


# Step 5: Train Model
print("\n Starting Training...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path)

model.info()

model.train(
    data=yaml_output_path,
    task="segment",
    device=device,
    epochs=epochs,
    batch=batch_size,
    project="cgras_segmentation",
    workers=workers,
    patience=patience,
    pretrained=pretrained,
    save_period=save_period,
    deterministic=False,
    imgsz=640,
    scale=0.2,
    flipud=0.5,
    fliplr=0.5
)

print("\n Training complete!")
