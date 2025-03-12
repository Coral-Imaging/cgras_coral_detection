#! /usr/bin/env python3

""" trian_segmenter.py
    Train basic model for image segmentation.
    This requires that the images have been tiled and split into train/val/test datasets.
    If not, please use the train_pipeline.py script to process the datasets.
"""

import torch
from ultralytics import YOLO
import torch


# Training parameters
# model_path = "yolov8n-seg.pt"
model_path = "/home/java/data/best.pt"
yaml_data_path = "/home/java/data/outputs/cgras_data.yaml"
pretrained = True
epochs = 16
batch_size = 16
workers = 4
save_period = 2
patience = 4

print("\n Starting YOLOv8 Training...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = YOLO(model_path)

model.info()

model.train(
    data=yaml_data_path,
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