#!/usr/bin/env python3

import os

# define where image files are located
# recursively search for all image files in the target directory
# check each image if it is a valid jpg image file
# report which images are valid, and which are not in two different text files
# save the output of each valid and invalid images in their respective text files



import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import logging

def is_valid_jpg(file_path):
    """
    Check if a file is a valid JPG image.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        bool: True if the image is a valid JPG, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            
            # Verify it's a JPG format
            # if img.format not in ['.jpg', '.jpeg', 'JPEG', 'JPG']:
            #     return False
            
            # Force load the image data to check for corruption
            img.load()
            
        return True
    except Exception as e:
        logging.debug(f"Error validating {file_path}: {e}")
        return False

def find_image_files(directory):
    """
    Recursively find all files with image extensions in the directory.
    
    Args:
        directory (str): Directory to search
        
    Returns:
        list: Paths to all image files found
    """
    image_files = []
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Check for valid JPG images in a directory")
    parser.add_argument("--dir", type=str, default='/home/dtsai/Data/cgras_datasets/cgras_amag_2024_highdensityexperiment', help="Target directory to search (default: current working directory)")
    parser.add_argument("--valid-output", type=str, default="valid_images.txt",
                        help="Output file for valid images (default: valid_images.txt)")
    parser.add_argument("--invalid-output", type=str, default="invalid_images.txt",
                        help="Output file for invalid images (default: invalid_images.txt)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(message)s')
    
    # Check if directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist")
        return 1
        
    print(f"Searching for image files in: {args.dir}")
    
    # Find all image files
    image_files = find_image_files(args.dir)
    print(f"Found {len(image_files)} image files")
    
    # Validate each image
    valid_images = []
    invalid_images = []
    
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"Checking image {i+1}/{len(image_files)}...")
            
        if is_valid_jpg(image_file):
            valid_images.append(image_file)
        else:
            invalid_images.append(image_file)
    
    # Write results to output files
    with open(args.valid_output, 'w') as f:
        f.write(f"Valid JPG Images ({len(valid_images)}):\n")
        for img in valid_images:
            f.write(f"{img}\n")
    
    with open(args.invalid_output, 'w') as f:
        f.write(f"Invalid or Corrupted JPG Images ({len(invalid_images)}):\n")
        for img in invalid_images:
            f.write(f"{img}\n")
    
    # Print summary
    print(f"\nResults:")
    print(f"  Valid JPG images: {len(valid_images)}")
    print(f"  Invalid/corrupted JPG images: {len(invalid_images)}")
    print(f"\nValid images saved to: {args.valid_output}")
    print(f"Invalid images saved to: {args.invalid_output}")
    
    if len(invalid_images) > 0:
        print("\nWARNING: Some invalid or corrupted JPG files were found!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())