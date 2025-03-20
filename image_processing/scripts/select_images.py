#! /usr/bin/env python3

""" select_all_images.py
    A simple script to select all images using the ImageSelector class.
    All parameters are hardcoded at the top of the script for easy configuration.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_selector import ImageSelector


# Configuration parameters - modify these as needed
SSD_PATH = "/media/java/RRAP03"  # Set your SSD path here
BASE_PATH = "cgras_2024_aims_camera_trolley"  # Base directory containing the image dataset
OUTPUT_PATH = "Pdae_10"  # Output directory name (will be created under SSD_PATH/outputs/)

# Selection parameters - set to 'all' to include everything
NUM_IMAGES = '10'  # Set to 'all' to select all matching images
SPAWN_MONTHS = 'all'  # Set to 'all' or a list like ['nov', 'oct']
WEEKS = 'all'  # Set to 'all' or specific weeks
SPECIES = 'all'  # Set to 'all' or specific species (e.g., 'Amag')
TILE_NUMBERS = 'all'  # Set to 'all' or specific tile numbers
POSITIONS = 'all'  # Set to 'all' or specific position indices

def main():
    print(f"Selecting all images from {os.path.join(SSD_PATH, BASE_PATH)}")
    print(f"Output directory: {os.path.join(SSD_PATH, 'outputs', OUTPUT_PATH)}")
    
    num_images_param = NUM_IMAGES
    if NUM_IMAGES != 'all':
        num_images_param = int(NUM_IMAGES)

    # Create ImageSelector instance with the configured parameters
    selector = ImageSelector(
        ssd_path=SSD_PATH,
        base_path=BASE_PATH,
        output_path=OUTPUT_PATH,
        num_images=num_images_param,
        spawn_months=SPAWN_MONTHS,
        weeks=WEEKS,
        species=SPECIES,
        tile_numbers=TILE_NUMBERS,
        positions=POSITIONS
    )
    
    # Select and copy images
    selected_images = selector.select_images()
    
    print(f"Successfully selected and copied {len(selected_images)} images")
    print(f"Images copied to: {selector.output_path}")

if __name__ == "__main__":
    main()