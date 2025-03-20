#! /usr/bin/env python3

""" trian.py
    Train basic model for image segmentation.
    This requires that the images have been tiled and split into train/val/test datasets.
    If not, please use the train_pipeline.py script to process the datasets.
"""
import os
import torch

from ultralytics import YOLO

SSD_PATH = "/media/java/RRAP03"

# Training parameters
MODEL_NAME = "best_2024.pt"
# MODEL_NAME = "yolov8n-seg.pt"

PROJECT = "cgras_segmentation"
NAME = "train_polyp"
CLASSES = [0,1] # Polyp
# NAME = "train_coral"
# CLASSES = [2,3] # Coral
# NAME = "train_coral_polyp"
# CLASSES = [0,1,2,3] # Coral and Polyp

PRETRAINED = True
EPOCHS = 16
BATCH_SIZE = 16
WORKERS = 4
SAVE_PERIOD = 2
PATIENCE = 4

model_path = os.path.join(SSD_PATH, "models", MODEL_NAME)
yaml_data_path = os.path.join(SSD_PATH, "outputs/train/cgras_data.yaml")

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
    imgsz=640,
    scale=0.2,
    flipud=0.5,
    fliplr=0.5
)

print("\n Training complete!")