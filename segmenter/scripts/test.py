#!/usr/bin/env python3
"""
Script to run the YOLOv8 segmentation tester on a model and dataset.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tester import SegmentationTester

# MODEL_PATH = "/home/java/repos/cgras_coral_detection/segmenter/cgras_segmentation/train7/weights/best.pt"
MODEL_PATH = "/home/reggie/hpc-home/models/cgras_data_2025Pdae_trained_20250524_modelyolo8xseg.pt"
YAML_PATH = "/home/reggie/hpc-home/Data/cgras/aspa/dataset_2023_built_from_testSet_122/cgras_2023+2024_dataset_seg_20250326.yaml"
OUTPUT_PATH = "/home/reggie/hpc-home/outputs/pdae_test"
CONF_THRESHOLD = 0.5

def main(model, yaml, output, conf):
    """Test a YOLOv8 segmentation model and visualize results."""
    print(f"Initializing segmentation tester with:")
    print(f"  Model: {model}")
    print(f"  YAML config: {yaml}")
    print(f"  Output path: {output}")
    print(f"  Confidence threshold: {conf}")
    
    # Initialize and run the segmentation tester
    tester = SegmentationTester(
        model_path=model,
        yaml_path=yaml,
        output_path=output,
        conf_threshold=conf
    )
    
    # Run predictions on all test images
    tester.run_predictions()
    
    # Generate summary of the test results
    tester.generate_summary()

if __name__ == "__main__":
    main(MODEL_PATH, YAML_PATH, OUTPUT_PATH, CONF_THRESHOLD)