#Script designed to combine two coco datasets into one replacing labels of the base dataset with labels from the additional dataset on certain images 
#Input: One base coco dataset and one additional coco dataset to overwrite the base dataset on certain images
#Output: A new coco dataset with the images from the base dataset and the labels from the additional dataset

import json
import argparse
import os
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Combine two COCO datasets into one')
    parser.add_argument('--base', required=False, default="/home/reggie/hpc-home/Data/cgras/amil/amil_predict_100/fixed_amil_predict.json" , help='Path to base COCO JSON file')
    parser.add_argument('--additional', required=False,default="/home/reggie/Downloads/instances_default.json" ,help='Path to additional COCO JSON file whose labels will override base dataset')
    parser.add_argument('--output', required=False,default="/home/reggie/hpc-home/Data/cgras/amil/amil_predict_100.json" ,help='Path to output combined COCO JSON file')
    return parser.parse_args()

def load_coco_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def combine_coco_datasets(base_data, additional_data):
    # Create image filename to ID mapping for both datasets
    base_img_map = {os.path.basename(img['file_name']): img['id'] for img in base_data['images']}
    additional_img_map = {os.path.basename(img['file_name']): img['id'] for img in additional_data['images']}
    
    # Find common images (those that exist in both datasets)
    common_img_filenames = set(base_img_map.keys()) & set(additional_img_map.keys())
    print(f"Found {len(common_img_filenames)} common images between datasets")

    # Create a new dataset with base structure
    combined_data = {
        'info': base_data.get('info', {}),
        'licenses': base_data.get('licenses', []),
        'images': base_data['images'],
        'categories': []
    }
    
    # Combine categories (avoiding duplicates by name)
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
    
    # Process annotations
    # Start by collecting annotations by image ID
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
    
    combined_data['annotations'] = combined_annotations
    return combined_data

def main():
    args = parse_args()
    
    # Load datasets
    base_data = load_coco_data(args.base)
    additional_data = load_coco_data(args.additional)
    
    # Combine datasets
    combined_data = combine_coco_datasets(base_data, additional_data)
    
    # Save combined dataset
    with open(args.output, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined dataset saved to {args.output}")
    print(f"Total images: {len(combined_data['images'])}")
    print(f"Total annotations: {len(combined_data['annotations'])}")
    print(f"Total categories: {len(combined_data['categories'])}")

if __name__ == '__main__':
    main()
