#!/usr/bin/env python3
"""
Fix COCO JSON by removing invalid annotations (those with empty bbox or segmentation)
"""

import json
import os
import sys
from datetime import datetime

def fix_coco_json(input_file, output_file=None):
    """
    Fix a COCO JSON file by removing annotations with empty bbox or segmentation.
    
    Args:
        input_file (str): Path to the input COCO JSON file
        output_file (str, optional): Path to the output COCO JSON file. If None, will use input filename with "_fixed" suffix.
    
    Returns:
        str: Path to the fixed COCO JSON file
    """
    # Set default output file name if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_fixed{ext}"
    
    # Read the input file
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
    
    # Count original annotations
    orig_count = len(coco_data['annotations'])
    print(f"Original COCO file has {orig_count} annotations")
    
    # Filter out invalid annotations (those with empty bbox or segmentation)
    valid_annotations = []
    removed_count = 0
    
    for ann in coco_data['annotations']:
        if not ann['bbox'] or not ann['segmentation']:
            print(f"Removing invalid annotation ID {ann['id']} (empty bbox or segmentation)")
            removed_count += 1
        else:
            valid_annotations.append(ann)
    
    # Update annotations in the COCO data
    coco_data['annotations'] = valid_annotations
    
    # Update the date_created field in the info section if it exists
    if 'info' in coco_data and 'date_created' in coco_data['info']:
        coco_data['info']['date_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write the fixed data to the output file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"Fixed COCO file has {len(valid_annotations)} annotations ({removed_count} removed)")
    print(f"Saved fixed COCO JSON to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Check if input file is provided as command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        fix_coco_json(input_file, output_file)
    else:
        print("Usage: python fix_coco_json.py <input_json_file> [output_json_file]")
        sys.exit(1)