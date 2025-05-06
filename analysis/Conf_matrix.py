import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from datetime import datetime

# Import from existing modules
from NegDataimages import Detector, ImageProcessor, DatasetAnalyzer, Prediction

# Import from new modules
from conf_matrix_script.confusion_matrix_analyzer import ConfusionMatrixAnalyzer
from conf_matrix_script.config_utils import load_config, get_default_config_path

CONFIG = "confusion_matrix_config.yaml"
default_config = get_default_config_path(CONFIG)

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
        
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Confusion Matrix Analysis')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to the configuration file')
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        return
    
    # Extract configuration values
    test_img_folder = config['dataset']['test_img_folder']
    test_label_folder = config['dataset']['test_label_folder']
    model_weights = config['model']['model_weights']
    output_dir = config['output']['output_dir']
    max_images = config['parameters']['max_images']
    min_area = config['parameters']['min_area']
    plot_normalized = config['visualization']['plot_normalized']
    plot_raw = config['visualization']['plot_raw']
    
    # Get classes and class colors from config instead of Utils
    classes = config.get('classes', ["alive_coral", "dead_coral"])
    
    # Create the class_colours dictionary from config colors
    class_colours = {}
    if 'colors' in config and 'class_colours' in config['colors']:
        class_colours = config['colors']['class_colours']
    else:
        # Fallback to default colors if not specified in config
        default_colors = {
            'blue': [0, 212, 255],
            'green': [0, 255, 0]
        }
        for i, class_name in enumerate(classes):
            if i == 0:
                class_colours[class_name] = default_colors['blue']
            elif i == 1:
                class_colours[class_name] = default_colors['green']
    
    print("Starting Confusion Matrix Analysis")
    print(f"Using config from: {args.config}")
    print(f"Classes: {classes}")
    print(f"Class colors: {class_colours}")
    
    # Initialize classes
    detection_type = config.get('detection_type', 'segmentation')
    print(f"Detection type: {detection_type}")
    detector = Detector(model_weights)
    img_processor = ImageProcessor(classes, class_colours)
    dataset_analyzer = DatasetAnalyzer(img_processor)
    
    # Initialize the confusion matrix analyzer - pass class_colours
    cm_analyzer = ConfusionMatrixAnalyzer(detector, dataset_analyzer, classes, class_colours, detection_type)    
    
    # Get list of test images and labels
    img_list = sorted(glob.glob(os.path.join(test_img_folder, '*.jpg')))
    label_list = sorted(glob.glob(os.path.join(test_label_folder, '*.txt')))
    
    # Limit number of images for processing
    img_list = img_list[:max_images]
    label_list = label_list[:max_images]
    
    print(f"Processing {len(img_list)} images with minimum area threshold: {min_area}")
    
    # Process dataset and collect labels
    cm_analyzer.process_dataset(img_list, label_list, min_area)
    
    # Create and plot confusion matrix
    if plot_normalized:
        cm_analyzer.plot_confusion_matrix(output_dir, normalize="row")
    
    if plot_raw:
        cm_analyzer.plot_confusion_matrix(output_dir, normalize=None)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()