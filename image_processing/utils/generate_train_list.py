#!/usr/bin/env python3

import os
import yaml
import argparse
from pathlib import Path

def generate_train_list(yaml_path, output_dir):
    """Generate train.txt with relative paths to images."""
    # Load YAML file
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Get base path for making paths relative
    base_path = Path(os.path.dirname(yaml_path))
    
    # Extract image paths for train split
    train_images = []
    if 'train' in data:
        for item in data['train']:
            img_path = Path(item['image'])
            # Make path relative to YAML file location
            relative_path = os.path.relpath(img_path, base_path)
            train_images.append(relative_path)

    # Write train.txt
    output_path = Path(output_dir) / 'train.txt'
    with open(output_path, 'w') as f:
        for img_path in sorted(train_images):  # Sort for consistency
            f.write(f"{img_path}\n")
    
    print(f"Generated train.txt with {len(train_images)} image paths at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate train.txt with relative image paths')
    parser.add_argument('--yaml', required=True, help='Path to YAML file containing dataset info')
    parser.add_argument('--output', required=True, help='Output directory for train.txt')
    
    args = parser.parse_args()
    generate_train_list(args.yaml, args.output)