#!/usr/bin/env python3
import os
import sys
import json
import logging
import argparse
from datetime import datetime

from PIL import Image
from pathlib import Path
from tqdm import tqdm

from sahi import AutoDetectionModel
from sahi.prediction import PredictionResult
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation

# Import SAHI components
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import list_files_with_extensions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run SAHI prediction with COCO format output")
    parser.add_argument("--project", type=str, default="sahi_project", help="Project name")
    parser.add_argument("--name", type=str, default="default", help="Experiment name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for annotations")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLOv8 model weights")
    parser.add_argument("--slice_width", type=int, default=640, help="Width of slices")
    parser.add_argument("--slice_height", type=int, default=640, help="Height of slices")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio for slicing")
    parser.add_argument("--conf_thresh", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    return parser.parse_args()

def main():
    # Parse arguments or use defaults
    try:
        args = parse_arguments()
        PROJECT = args.project
        NAME = args.name
        DATA_DIR = args.data_dir
        OUTPUT_DIR = args.output_dir
        MODEL_PATH = args.model_path
        SLICE_WIDTH = args.slice_width
        SLICE_HEIGHT = args.slice_height
        OVERLAP = args.overlap
        CONF_THRESH = args.conf_thresh
        DEVICE = args.device
    except:
        # Use defaults if run without argparse
        PROJECT = "sahi_project"
        NAME = "Pdae_100"
        DATA_DIR = "/media/wardlewo/RRAP02/cgras_pdae_2024_aims/pdae_100"
        OUTPUT_DIR = "/home/wardlewo/Reggie/data"
        MODEL_PATH = "/home/wardlewo/hpc-home/runs/pdae_29042025_cgras_seg_first_30/20250326_8n_train_multiGpu_B128/weights/best.pt"
        SLICE_WIDTH = 640
        SLICE_HEIGHT = 640
        OVERLAP = 0.5
        CONF_THRESH = 0.4
        DEVICE = "cuda:0"

    ### VALIDATION ###
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Data directory does not exist: {DATA_DIR}")
    
    # Create output directory structure
    annotations_dir = os.path.join(OUTPUT_DIR, NAME + "_annotations_" + 
                                  datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_coco 1.0")
    os.makedirs(os.path.join(annotations_dir, "annotations"), exist_ok=True)
    
    # Get list of images
    image_files = list_files_with_extensions(
        directory=DATA_DIR,
        extensions=[".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    )
    print(f"Number of images found: {len(image_files)}")

    ### MODEL INITIALIZATION ###
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESH,
        device=DEVICE
    )

    # Initialize COCO object with proper structure
    coco = Coco()
    
    # Define categories
    categories = ["alive_coral", "dead_coral"]
    category_mapping = {}
    
    for i, category_name in enumerate(categories, 1):
        category = CocoCategory(id=i, name=category_name)
        coco.add_category(category)
        category_mapping[category_name] = i

    ### PROCESS IMAGES AND CREATE ANNOTATIONS ###
    annotation_id = 1  # Initialize annotation ID counter
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
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
            
            # Get sliced predictions with error handling
            try:
                # Disable postprocess to avoid segmentation mask errors
                result = get_sliced_prediction(
                    image=image_path,
                    detection_model=detection_model,
                    slice_height=SLICE_HEIGHT,
                    slice_width=SLICE_WIDTH,
                    overlap_height_ratio=OVERLAP,
                    overlap_width_ratio=OVERLAP,
                    perform_standard_pred=False,
                    postprocess_type="NMS",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.2,
                    verbose=0
                )
            except ValueError as e:
                logger.warning(f"Sliced prediction with NMS failed for {image_filename}: {e}")
                try:
                    # Try directly with detection model, no slice
                    detection_result = detection_model.predict(image=image_path)
                    result = PredictionResult(
                        image=image_path,
                        object_prediction_list=detection_result.object_prediction_list
                    )
                    logger.info(f"Used direct prediction for {image_filename}")
                except Exception as direct_err:
                    logger.error(f"Direct prediction failed for {image_filename}: {direct_err}")
                    # Create empty result to continue processing
                    result = PredictionResult(image=image_path, object_prediction_list=[])
            
            # Process each prediction and create COCO annotations
            for pred in result.object_prediction_list:
                try:
                    predicted_category_name = pred.category.name
                    if predicted_category_name in category_mapping:
                        category_id = category_mapping[predicted_category_name]
                    else:
                        logger.warning(f"Unknown category '{predicted_category_name}', using default")
                        category_id = 1
                        predicted_category_name = categories[0]

                    # Get bounding box in COCO format [x, y, width, height]
                    x = pred.bbox.minx
                    y = pred.bbox.miny
                    box_width = pred.bbox.maxx - pred.bbox.minx
                    box_height = pred.bbox.maxy - pred.bbox.miny
                    
                    # Skip annotations with zero width or height to avoid CVAT errors
                    if box_width <= 0 or box_height <= 0:
                        logger.warning(f"Skipping invalid bounding box with zero width/height")
                        continue
                        
                    bbox = [x, y, box_width, box_height]
                    
                    # Calculate area
                    area = box_width * box_height
                    
                    # Handle segmentation safely
                    try:
                        if hasattr(pred, 'mask') and pred.mask is not None and hasattr(pred.mask, 'segmentation'):
                            segmentation = pred.mask.segmentation
                            # Validate segmentation format
                            if not isinstance(segmentation, list) or len(segmentation) == 0:
                                raise ValueError("Empty segmentation list")
                        else:
                            raise AttributeError("No valid mask found")
                    except (AttributeError, ValueError) as e:
                        # Create segmentation from bbox as fallback
                        logger.debug(f"Using bbox for segmentation due to: {e}")
                        segmentation = [[
                            x, y, 
                            x + box_width, y,
                            x + box_width, y + box_height,
                            x, y + box_height
                        ]]
                    
                    # Skip if segmentation is empty or invalid
                    if not segmentation or not all(isinstance(s, list) and len(s) > 0 for s in segmentation):
                        logger.warning(f"Skipping annotation with invalid segmentation")
                        continue
                        
                    # Create COCO annotation without the problematic parameters
                    coco_annotation = CocoAnnotation(
                        bbox=bbox,
                        category_id=category_id,
                        category_name=predicted_category_name,
                        image_id=coco_image.id,
                        segmentation=segmentation,
                        iscrowd=0
                    )
                    
                    # Set ID and area manually after creation
                    coco_annotation.id = annotation_id
                    # Add area to the json representation of the annotation
                    coco_annotation.json["area"] = area
                    annotation_id += 1
                    
                    # Add annotation to image
                    coco_image.add_annotation(coco_annotation)
                except Exception as pred_err:
                    logger.error(f"Error processing prediction for {image_filename}: {pred_err}")
                    continue
            
            # Add the image to COCO object
            coco.add_image(coco_image)
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue

    # Ensure COCO info structure matches the expected format
    coco_json = coco.json
    coco_json["info"] = {
        "description": f"COCO-formatted annotations for {NAME}",
        "url": "",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "SAHI automatic annotation",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add licenses section
    coco_json["licenses"] = [
        {
            "url": "",
            "id": 1,
            "name": "Unknown"
        }
    ]

    ### SAVE COCO JSON ###
    output_path = os.path.join(annotations_dir, "annotations", "instances_default.json")
    with open(output_path, "w") as f:
        json.dump(coco_json, f)

    print(f"COCO annotations saved to: {output_path}")
    print(f"Total images processed: {len(coco.images)}")

if __name__ == "__main__":
    main()