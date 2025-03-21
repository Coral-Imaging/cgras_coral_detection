#! /usr/bin/env python3
""" resume_train.py
 Resume training for a YOLOv8 image segmentation model.
 This script picks up from the last checkpoint saved during a previous training run.
"""
import os
import glob
import torch
from ultralytics import YOLO

# Path configurations - keep the same as original training
DATA_PATH = "/mnt/hpccs01/home/gonia/data"
# DATA_PATH = "/media/java/RRAP03"

# Training parameters - keep the same as original training
PROJECT = "cgras_segmentation"
NAME = "train_coral_polyp2" # normally remove the 2
CLASSES = [0, 1, 2, 3]  # Coral and Polyp
PRETRAINED = True  # This won't matter when resuming
EPOCHS = 750  # Total epochs (including previously run ones)
BATCH_SIZE = 32
WORKERS = 16
SAVE_PERIOD = 10
PATIENCE = 20
MASK_OVERLAP = False

yaml_data_path = os.path.join(DATA_PATH, "train/cgras_data.yaml")

weights_dir = os.path.join(PROJECT, NAME, "weights")
best_model = os.path.join(weights_dir, "best.pt")

if not os.path.exists(best_model):
    best_model = os.path.join(weights_dir, "last.pt") # if best.pt doesn't exist, use last.pt

print(f"Extending training from model: {best_model}")

# Resume the training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load the model from the latest checkpoint
model = YOLO(best_model)
model.info()

# Resume training with the same parameters
model.train(
    data=yaml_data_path,
    task="segment",
    device=device,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    project=PROJECT,
    name=NAME + "_resume",
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
    fliplr=0.5,
    resume=True
)

print("\n Training resumed and completed!")