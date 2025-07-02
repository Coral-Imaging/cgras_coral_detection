#!/usr/bin/env python3
"""
Script to run the YOLOv8 segmentation validator on a model and dataset.
Compares model predictions with ground truth labels.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.validator import SegmentationValidator

# MODEL_PATH = "/home/java/repos/cgras_coral_detection/segmenter/cgras_segmentation/train7/weights/best.pt"
MODEL_PATH = "/home/alexanderjones/Alex/hpc-home/data/amag/model_outputs/train_amag_early10/weights/best.pt"
YAML_PATH = "/home/alexanderjones/Alex/hpc-home/data/amag/segmentation_outputs/amag_early10_filtered_split_tiled_balanced/cgras_data.yaml"
OUTPUT_PATH = "/home/alexanderjones/Alex/hpc-home/data/amag/prediction_outputs"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

def main(model, yaml, output, conf, iou):
    """Validate a YOLOv8 segmentation model by comparing predictions with ground truth labels."""
    print(f"Initializing segmentation validator with:")
    print(f"  Model: {model}")
    print(f"  YAML config: {yaml}")
    print(f"  Output path: {output}")
    print(f"  Confidence threshold: {conf}")
    print(f"  IoU threshold: {iou}")
    
    # Initialize and run the segmentation validator
    validator = SegmentationValidator(
        model_path=model,
        yaml_path=yaml,
        output_path=output,
        conf_threshold=conf,
        iou_threshold=iou
    )
    
    # Run validation on all validation images
    validator.run_validation()
    
    # Generate summary of the validation results
    validator.generate_summary()

if __name__ == "__main__":
    main(MODEL_PATH, YAML_PATH, OUTPUT_PATH, CONF_THRESHOLD, IOU_THRESHOLD)