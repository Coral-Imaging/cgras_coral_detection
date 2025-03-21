#! /usr/bin/env python3

""" trian.py
    Train basic model for image segmentation.
    This requires that the images have been tiled and split into train/val/test datasets.
    If not, please use the train_pipeline.py script to process the datasets.
"""
import os
import torch

from ultralytics import YOLO

DATA_PATH = "/mnt/hpccs01/home/gonia/data"
# DATA_PATH = "/media/java/RRAP03"

# Training parameters
MODEL_NAME = "best_2024.pt"

PROJECT = "cgras_segmentation"
NAME = "train_coral_polyp"
CLASSES = [0,1,2,3] # Coral and Polyp

PRETRAINED = True
EPOCHS = 1000
BATCH_SIZE = 0.8
WORKERS = 32
SAVE_PERIOD = 10
PATIENCE = 25
MASK_OVERLAP = False

model_path = os.path.join(DATA_PATH, "models", MODEL_NAME)
yaml_data_path = os.path.join(DATA_PATH, "train/cgras_data.yaml")

# Train the model
print("\n Starting YOLOv8 Training...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = YOLO(model_path)
model.info()

model.train(
    data=yaml_data_path,
    task="segment",
    device=device,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    project=PROJECT,
    name=NAME,
    classes=CLASSES,
    workers=WORKERS,
    patience=PATIENCE,
    pretrained=PRETRAINED,
    save_period=SAVE_PERIOD,
    deterministic=False,
    overlap_mask=MASK_OVERLAP,
    imgsz=640,
    scale=0.2,
    flipud=0.5,
    fliplr=0.5
)

print("\n Training complete!")