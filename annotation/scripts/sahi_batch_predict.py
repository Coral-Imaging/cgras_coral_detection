#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from sahi.predict import predict
from sahi import AutoDetectionModel

# Import your utility functions
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import create_temp_coco_json, copy_coco_results

### PARAMETERS ###
PROJECT = "sahi_project"
NAME = "test"
DATA_DIR = "/media/java/RRAP03/outputs/test/sahi/"  # Replace with your image folder path
OUTPUT_DIR = "/media/java/RRAP03/outputs/test/coco_annotations/"  # Replace with your desired output folder path
MODEL_PATH = "/home/java/hpc-home/cgras_segmentation/train_coral_polyp2/weights/best.pt"  # Replace with your model path
SLICE_WIDTH = 640
SLICE_HEIGHT = 640
OVERLAP = 0.5
CONF_THRESH = 0.5
DEVICE = "cuda:0"  # or 'cpu'

### VALIDATION ###
if not os.path.exists(DATA_DIR):
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Number of images in directory: {len(os.listdir(DATA_DIR))}")

### CREATE TEMPORARY COCO JSON ###
# Using the utility function to create temporary COCO JSON
temp_coco_path = create_temp_coco_json(DATA_DIR)
print(f"Created temporary COCO JSON at: {temp_coco_path}")

### RUN PREDICTION ###
# Load the model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRESH,
    device=DEVICE
)

# Run prediction with COCO export
results = predict(
    project=PROJECT,
    name=NAME,
    export_visuals=True,
    detection_model=detection_model,
    source=DATA_DIR,
    dataset_json_path=temp_coco_path,
    slice_height=SLICE_HEIGHT,
    slice_width=SLICE_WIDTH,
    overlap_height_ratio=OVERLAP,
    overlap_width_ratio=OVERLAP,
    verbose=2,
    return_dict=True
)

# Check if results is None
if results is None:
    raise ValueError("Prediction results are None. Please check the predict function for errors.")