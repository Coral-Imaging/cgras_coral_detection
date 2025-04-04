#! /usr/bin/env python3

""" train.py
    Train basic model for image segmentation.
    This requires that the images have been tiled and split into train/val/test datasets.
    If not, please use the train_pipeline.py script to process the datasets.
"""

import argparse
import yaml
import os
from ultralytics import YOLO

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation model with YAML configuration.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def train_model(config):
    """Train YOLOv8 model with the provided configuration."""
    print("\nStarting YOLOv8 Training...")
    
    # Expand user path if needed (for ~/ paths)
    model_path = os.path.expanduser(config['model_path'])
    
    # Initialize model
    model = YOLO(model_path)
    model.info()
    
    # Train the model
    model.train(
        data=config['yaml_path'],
        task="segment",
        device=config['device'],
        epochs=config['epochs'],
        batch=config['batch_size'],
        project=config['project'],
        name=config['name'],
        classes=config['classes'],
        workers=config['workers'],
        patience=config['patience'],
        pretrained=config['pretrained'],
        save_period=config['save_period'],
        deterministic=False,
        overlap_mask=config['mask_overlap'],
        imgsz=config['image_size'],
        scale=config['scale'],
        flipud=config['flipud'],
        fliplr=config['fliplr']
    )
    
    print("\nTraining complete!")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train model with configuration
    train_model(config)

if __name__ == "__main__":
    main()