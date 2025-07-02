#!/usr/bin/env python3
"""
Fix COCO JSON by removing invalid annotations (those with empty bbox or segmentation)
"""

import json
import os
import sys
from datetime import datetime


class CocoJsonFixer:
    """
    A class to fix COCO JSON files by removing invalid annotations.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the CocoJsonFixer.
        
        Args:
            verbose (bool): Whether to print progress messages
        """
        self.verbose = verbose
    
    def fix_coco_json(self, input_file, output_file=None):
        """
        Fix a COCO JSON file by removing annotations with empty bbox or segmentation.
        
        Args:
            input_file (str): Path to the input COCO JSON file
            output_file (str, optional): Path to the output COCO JSON file. 
                                       If None, will use input filename with "_fixed" suffix.
        
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
        if self.verbose:
            print(f"Original COCO file has {orig_count} annotations")
        
        # Filter out invalid annotations
        valid_annotations, removed_count = self._filter_invalid_annotations(coco_data['annotations'])
        
        # Update annotations in the COCO data
        coco_data['annotations'] = valid_annotations
        
        # Update the date_created field in the info section if it exists
        self._update_date_created(coco_data)
        
        # Write the fixed data to the output file
        with open(output_file, 'w') as f:
            json.dump(coco_data, f)
        
        if self.verbose:
            print(f"Fixed COCO file has {len(valid_annotations)} annotations ({removed_count} removed)")
            print(f"Saved fixed COCO JSON to: {output_file}")
        
        return output_file
    
    def _filter_invalid_annotations(self, annotations):
        """
        Filter out invalid annotations (those with empty bbox or segmentation).
        
        Args:
            annotations (list): List of annotation dictionaries
            
        Returns:
            tuple: (valid_annotations, removed_count)
        """
        valid_annotations = []
        removed_count = 0
        
        for ann in annotations:
            if not ann['bbox'] or not ann['segmentation']:
                if self.verbose:
                    print(f"Removing invalid annotation ID {ann['id']} (empty bbox or segmentation)")
                removed_count += 1
            else:
                valid_annotations.append(ann)
        
        return valid_annotations, removed_count
    
    def _update_date_created(self, coco_data):
        """
        Update the date_created field in the info section if it exists.
        
        Args:
            coco_data (dict): COCO data dictionary
        """
        if 'info' in coco_data and 'date_created' in coco_data['info']:
            coco_data['info']['date_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_annotation_stats(self, input_file):
        """
        Get statistics about annotations in a COCO file without modifying it.
        
        Args:
            input_file (str): Path to the input COCO JSON file
            
        Returns:
            dict: Statistics about the annotations
        """
        with open(input_file, 'r') as f:
            coco_data = json.load(f)
        
        total_annotations = len(coco_data['annotations'])
        valid_annotations, removed_count = self._filter_invalid_annotations(coco_data['annotations'])
        
        return {
            'total_annotations': total_annotations,
            'valid_annotations': len(valid_annotations),
            'invalid_annotations': removed_count
        }


# Keep the original function for backward compatibility
def fix_coco_json(input_file, output_file=None):
    """
    Legacy function wrapper for backward compatibility.
    """
    fixer = CocoJsonFixer()
    return fixer.fix_coco_json(input_file, output_file)


if __name__ == "__main__":
    # Check if input file is provided as command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        fixer = CocoJsonFixer()
        fixer.fix_coco_json(input_file, output_file)
    else:
        print("Usage: python fix_coco_json.py <input_json_file> [output_json_file]")
        sys.exit(1)