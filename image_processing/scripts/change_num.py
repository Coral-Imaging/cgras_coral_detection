#!/usr/bin/env python3
import os
import yaml
import glob
from pathlib import Path


def remap_annotation_file(file_path):
    """
    Remap classes in a YOLO annotation file:
    - Class 0 -> Class 2
    - Class 1 -> Class 3
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        parts = line.strip().split(' ', 1)  # Split only at the first space
        if len(parts) >= 2:
            class_id = int(parts[0])
            if class_id == 0:
                parts[0] = '2'
            elif class_id == 1:
                parts[0] = '3'
            modified_lines.append(' '.join(parts))
    
    with open(file_path, 'w') as f:
        for line in modified_lines:
            f.write(f"{line}\n")


def process_dataset_folders(base_path, folders):
    """Process all annotation files in the given folders"""
    for folder in folders:
        # Convert images path to labels path
        folder_path = os.path.join(base_path, folder)
        label_folder = folder_path.replace('/images', '/labels')
        
        # Handle the special case where it's /labels/labels
        if not os.path.exists(label_folder) and '/labels/images' in label_folder:
            label_folder = label_folder.replace('/labels/images', '/labels/labels')
        
        print(f"Processing annotations in: {label_folder}")
        
        # Find all txt files in the labels folder
        annotation_files = glob.glob(os.path.join(label_folder, '*.txt'))
        
        for file_path in annotation_files:
            remap_annotation_file(file_path)
            
        print(f"Processed {len(annotation_files)} annotation files")


def update_yaml_file(yaml_path):
    """Update the YAML file with dummy classes for 0 and 1"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update the names section to include dummy classes
    names = {}
    if 'names' in data:
        for key, value in data['names'].items():
            key_int = int(key)
            if key_int == 0:
                names[0] = 'dummy_class_0'
                names[2] = 'mask_live'
            elif key_int == 1:
                names[1] = 'dummy_class_1'
                names[3] = 'mask_dead'
            else:
                names[key_int] = value
    
    data['names'] = names
    
    # Write the updated YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Updated YAML file: {yaml_path}")


def main(yaml_path):
    """Main function to process the dataset"""
    print(f"Reading YAML file: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    base_path = data.get('path', '')
    print(f"Base path: {base_path}")
    
    # Process each dataset split
    for split in ['train', 'val', 'test']:
        if split in data:
            print(f"Processing {split} split...")
            process_dataset_folders(base_path, data[split])
    
    # Update the YAML file
    update_yaml_file(yaml_path)
    
    print("Class remapping completed successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python remap_classes.py <path_to_yaml_file>")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    main(yaml_path)