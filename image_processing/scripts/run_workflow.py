#!/usr/bin/env python3

"""
run_workflow.py

This script runs the complete data processing workflow:
1. Balance the dataset (remove excess empty images)
2. Split the balanced dataset into train/val/test
3. Tile the images within each split

Usage:
    python run_workflow.py --data_path "/media/java/RRAP03" --input "export100_from_cvat" --output "processed_dataset"

Options:
    --data_path      Base path for all operations
    --input         Input folder name (relative to data_path)
    --output        Output folder name (will be created under data_path/outputs)
    --tile_size     Tile size in pixels, default: 640x640
    --overlap       Overlap percentage between tiles, default: 50
    --train_split   Training split ratio, default: 0.7
    --val_split     Validation split ratio, default: 0.15
    --test_split    Test split ratio, default: 0.15
    --skip_balance  Skip the balance step if specified
    --verbose       Print detailed progress information
"""

import os
import sys
import argparse
import shutil
from process_dataset import main as process_dataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the complete data processing workflow.")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Base path for all operations")
    parser.add_argument("--input", type=str, required=True,
                        help="Input folder name (relative to data_path)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output folder name (will be created under data_path/outputs)")
    parser.add_argument("--tile_size", type=str, default="640,640",
                        help="Tile size in pixels (width,height), default: 640,640")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Overlap percentage between tiles, default: 50")
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Training split ratio, default: 0.7")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio, default: 0.15")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Test split ratio, default: 0.15")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 1e-10:
        parser.error("Split ratios must sum to 1.0")
    
    # Parse tile size
    try:
        width, height = map(int, args.tile_size.split(','))
        args.tile_size = (width, height)
    except:
        parser.error("Tile size must be in format 'width,height'")
    
    return args

def main():
    """Main function to run the workflow."""
    args = parse_args()
    
    # Update environment variables for the process_dataset script
    os.environ["DATA_PATH"] = args.data_path
    os.environ["INPUT_FOLDER"] = args.input
    os.environ["OUTPUT_FOLDER"] = args.output
    os.environ["TRAIN_SPLIT"] = str(args.train_split)
    os.environ["VAL_SPLIT"] = str(args.val_split)
    os.environ["TEST_SPLIT"] = str(args.test_split)
    os.environ["TILE_WIDTH"] = str(args.tile_size[0])
    os.environ["TILE_HEIGHT"] = str(args.tile_size[1])
    os.environ["OVERLAP"] = str(args.overlap)
    os.environ["VERBOSE"] = "True" if args.verbose else "False"
    
    # Run the process_dataset script
    process_dataset()

if __name__ == "__main__":
    main()