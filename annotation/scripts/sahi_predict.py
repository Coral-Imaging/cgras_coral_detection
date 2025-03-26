#!/usr/bin/env python3
import os
import sys
import json

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation

# Import SAHI components
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import list_files_with_extensions

### PARAMETERS ###
PROJECT = "sahi_project"
NAME = "test"
DATA_DIR = "/media/java/RRAP03/unlabelled/updated_2024_cgras_amag_TilesMix_first100_100quality/"  # Replace with your image folder path
OUTPUT_DIR = "/media/java/RRAP03/outputs/coco_annotations/"  # Replace with your desired output folder path
MODEL_PATH = "/home/java/hpc-home/cgras_segmentation/train_coral_polyp3/weights/best.pt"  # Replace with your model path
SLICE_WIDTH = 640
SLICE_HEIGHT = 640
OVERLAP = 0.5
CONF_THRESH = 0.4
DEVICE = "cuda:0"  # or 'cpu'

### VALIDATION ###
if not os.path.exists(DATA_DIR):
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get list of images
image_files = list_files_with_extensions(
    directory=DATA_DIR,
    extensions=[".jpg", ".jpeg", ".png", ".tif", ".tiff"]
)
print(f"Number of images found: {len(image_files)}")

### MODEL INITIALIZATION ###
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRESH,
    device=DEVICE
)

# Initialize COCO object with categories
coco = Coco()
categories = ["alive", "dead", "mask_live", "mask_dead"]
category_mapping = {}

for i, category_name in enumerate(categories, 1):
    coco.add_category(CocoCategory(id=i, name=category_name))
    category_mapping[category_name] = i


### PROCESS IMAGES AND CREATE ANNOTATIONS ###
for image_path in tqdm(image_files, desc="Processing images"):
    # Get image info
    image_filename = os.path.basename(image_path)
    image = Image.open(image_path)
    width, height = image.size
    
    # Create COCO image
    coco_image = CocoImage(
        file_name=image_filename,
        height=height,
        width=width,
    )
    
    # Get sliced predictions
    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        verbose=0
    )
    
    # Process each prediction and create COCO annotations
    for pred in result.object_prediction_list:
        predicted_category_name = pred.category.name
        if predicted_category_name in category_mapping:
            category_id = category_mapping[predicted_category_name]
        else:
            print(f"Warning: Unknown category '{predicted_category_name}', using default")
            category_id = 1
            predicted_category_name = categories[0]

        # Get bounding box in COCO format [x, y, width, height]
        x = pred.bbox.minx
        y = pred.bbox.miny
        width = pred.bbox.maxx - pred.bbox.minx
        height = pred.bbox.maxy - pred.bbox.miny
        bbox = [x, y, width, height]
        
        # Handle segmentation
        if hasattr(pred, 'mask') and pred.mask is not None:
            # If mask exists, convert to COCO segmentation format
            segmentation = pred.mask.segmentation
        else:
            # Create segmentation from bbox as fallback
            segmentation = [[
                x, y, 
                x + width, y,
                x + width, y + height,
                x, y + height
            ]]
        
        # Create COCO annotation
        coco_annotation = CocoAnnotation(
            bbox=bbox,
            category_id=category_id,
            category_name=predicted_category_name,
            image_id=coco_image.id,
            segmentation=segmentation,
        )
        
        # Add annotation to image
        coco_image.add_annotation(coco_annotation)
    
    # Add the image to COCO object
    coco.add_image(coco_image)

### SAVE COCO JSON ###
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"{NAME}_{timestamp}.json")
with open(output_path, "w") as f:
    json.dump(coco.json, f, indent=4)

print(f"COCO annotations saved to: {output_path}")
print(f"Total images processed: {len(coco.images)}")