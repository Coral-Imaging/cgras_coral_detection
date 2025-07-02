#!/usr/bin/env python3
"""
Script designed to combine two coco datasets into one replacing labels of the base dataset 
with labels from the additional dataset on certain images.

Input: One base coco dataset and one additional coco dataset to overwrite the base dataset on certain images
Output: A new coco dataset with the images from the base dataset and the labels from the additional dataset
"""

import json
import argparse
import os
from collections import defaultdict
from datetime import datetime


class CocoCombiner:
    """
    A class to combine two COCO datasets, replacing annotations from base dataset 
    with annotations from additional dataset for common images.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the CocoCombiner.
        
        Args:
            verbose (bool): Whether to print progress messages
        """
        self.verbose = verbose
    
    def load_coco_data(self, json_path):
        """
        Load COCO dataset from JSON file.
        
        Args:
            json_path (str): Path to COCO JSON file
            
        Returns:
            dict: COCO dataset dictionary
        """
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def combine_coco_datasets(self, base_data, additional_data):
        """
        Combine two COCO datasets, using additional dataset annotations for common images.
        
        Args:
            base_data (dict): Base COCO dataset
            additional_data (dict): Additional COCO dataset whose annotations override base
            
        Returns:
            dict: Combined COCO dataset
        """
        # Create image filename to ID mapping for both datasets
        base_img_map = {os.path.basename(img['file_name']): img['id'] for img in base_data['images']}
        additional_img_map = {os.path.basename(img['file_name']): img['id'] for img in additional_data['images']}
        
        # Find common images (those that exist in both datasets)
        common_img_filenames = set(base_img_map.keys()) & set(additional_img_map.keys())
        if self.verbose:
            print(f"Found {len(common_img_filenames)} common images between datasets")

        # Create a new dataset with base structure
        combined_data = {
            'info': base_data.get('info', {}),
            'licenses': base_data.get('licenses', []),
            'images': base_data['images'],
            'categories': []
        }
        
        # Update info section with combination timestamp
        if 'info' not in combined_data:
            combined_data['info'] = {}
        combined_data['info']['date_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        combined_data['info']['description'] = combined_data['info'].get('description', '') + ' [Combined dataset]'
        
        # Combine categories and create mapping
        seen_categories, additional_cat_id_to_combined = self._combine_categories(
            base_data, additional_data, combined_data
        )
        
        # Process annotations
        combined_annotations = self._combine_annotations(
            base_data, additional_data, base_img_map, additional_img_map, 
            common_img_filenames, additional_cat_id_to_combined
        )
        
        combined_data['annotations'] = combined_annotations
        return combined_data
    
    def _combine_categories(self, base_data, additional_data, combined_data):
        """
        Combine categories from both datasets, avoiding duplicates.
        
        Returns:
            tuple: (seen_categories dict, category ID mapping dict)
        """
        seen_categories = {}
        next_category_id = 1
        
        # Process base categories first
        for cat in base_data['categories']:
            cat_name = cat['name']
            if cat_name not in seen_categories:
                cat_copy = cat.copy()
                cat_copy['id'] = next_category_id
                combined_data['categories'].append(cat_copy)
                seen_categories[cat_name] = next_category_id
                next_category_id += 1
        
        # Add any new categories from the additional dataset
        for cat in additional_data['categories']:
            cat_name = cat['name']
            if cat_name not in seen_categories:
                cat_copy = cat.copy()
                cat_copy['id'] = next_category_id
                combined_data['categories'].append(cat_copy)
                seen_categories[cat_name] = next_category_id
                next_category_id += 1
        
        # Create category ID mapping for additional dataset
        additional_cat_id_to_combined = {}
        for cat in additional_data['categories']:
            additional_cat_id_to_combined[cat['id']] = seen_categories[cat['name']]
        
        return seen_categories, additional_cat_id_to_combined
    
    def _combine_annotations(self, base_data, additional_data, base_img_map, 
                           additional_img_map, common_img_filenames, additional_cat_id_to_combined):
        """
        Combine annotations from both datasets according to the replacement logic.
        
        Returns:
            list: Combined annotations list
        """
        # Process annotations - Start by collecting annotations by image ID
        base_annotations_by_img = defaultdict(list)
        for ann in base_data['annotations']:
            base_annotations_by_img[ann['image_id']].append(ann)
        
        additional_annotations_by_img = defaultdict(list)
        for ann in additional_data['annotations']:
            additional_annotations_by_img[ann['image_id']].append(ann)
        
        # Create combined annotations
        combined_annotations = []
        next_ann_id = 1
        
        # For each image in base dataset
        for img in base_data['images']:
            img_id = img['id']
            img_filename = os.path.basename(img['file_name'])
            
            # Check if image exists in additional dataset AND has annotations
            if img_filename in common_img_filenames:
                additional_img_id = additional_img_map[img_filename]
                
                # Only use additional annotations if they exist for this image
                if additional_annotations_by_img[additional_img_id]:
                    # Use annotations from additional dataset with converted category IDs
                    for ann in additional_annotations_by_img[additional_img_id]:
                        ann_copy = ann.copy()
                        ann_copy['id'] = next_ann_id
                        ann_copy['image_id'] = img_id
                        ann_copy['category_id'] = additional_cat_id_to_combined[ann['category_id']]
                        combined_annotations.append(ann_copy)
                        next_ann_id += 1
                else:
                    # If the image is in additional dataset but has no annotations,
                    # keep the annotations from the base dataset
                    for ann in base_annotations_by_img[img_id]:
                        ann_copy = ann.copy()
                        ann_copy['id'] = next_ann_id
                        combined_annotations.append(ann_copy)
                        next_ann_id += 1
            else:
                # Use annotations from base dataset
                for ann in base_annotations_by_img[img_id]:
                    ann_copy = ann.copy()
                    ann_copy['id'] = next_ann_id
                    combined_annotations.append(ann_copy)
                    next_ann_id += 1
        
        return combined_annotations
    
    def combine_datasets_from_files(self, base_path, additional_path, output_path):
        """
        Complete workflow to combine datasets from file paths.
        
        Args:
            base_path (str): Path to base COCO JSON file
            additional_path (str): Path to additional COCO JSON file
            output_path (str): Path to output combined COCO JSON file
            
        Returns:
            str: Path to the output file
        """
        # Load datasets
        base_data = self.load_coco_data(base_path)
        additional_data = self.load_coco_data(additional_path)
        
        # Combine datasets
        combined_data = self.combine_coco_datasets(base_data, additional_data)
        
        # Save combined dataset
        with open(output_path, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        if self.verbose:
            print(f"Combined dataset saved to {output_path}")
            print(f"Total images: {len(combined_data['images'])}")
            print(f"Total annotations: {len(combined_data['annotations'])}")
            print(f"Total categories: {len(combined_data['categories'])}")
        
        return output_path
    
    def get_dataset_stats(self, json_path):
        """
        Get statistics about a COCO dataset.
        
        Args:
            json_path (str): Path to COCO JSON file
            
        Returns:
            dict: Dataset statistics
        """
        data = self.load_coco_data(json_path)
        return {
            'images': len(data.get('images', [])),
            'annotations': len(data.get('annotations', [])),
            'categories': len(data.get('categories', []))
        }


# Keep the original functions for backward compatibility
def parse_args():
    """Legacy argument parser for backward compatibility."""
    parser = argparse.ArgumentParser(description='Combine two COCO datasets into one')
    parser.add_argument('--base', required=False, 
                       default="/home/wardlewo/Reggie/data/fix/instances_default.json", 
                       help='Path to base COCO JSON file')
    parser.add_argument('--additional', required=False,
                       default="//home/wardlewo/Reggie/data/job_2427841_annotations_2025_05_14_02_47_49_coco 1.0/annotations/instances_default.json",
                       help='Path to additional COCO JSON file whose labels will override base dataset')
    parser.add_argument('--output', required=False,
                       default="/home/wardlewo/Reggie/data/instances_default.json",
                       help='Path to output combined COCO JSON file')
    return parser.parse_args()


def load_coco_data(json_path):
    """Legacy function wrapper for backward compatibility."""
    combiner = CocoCombiner(verbose=False)
    return combiner.load_coco_data(json_path)


def combine_coco_datasets(base_data, additional_data):
    """Legacy function wrapper for backward compatibility."""
    combiner = CocoCombiner(verbose=False)
    return combiner.combine_coco_datasets(base_data, additional_data)


def main():
    """Main function using the class structure."""
    args = parse_args()
    
    combiner = CocoCombiner(verbose=True)
    combiner.combine_datasets_from_files(args.base, args.additional, args.output)


if __name__ == '__main__':
    main()