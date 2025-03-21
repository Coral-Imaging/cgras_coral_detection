#!/usr/bin/env python3
import os
import json
import shutil
import tempfile
from pathlib import Path

from typing import List, Union

def create_temp_coco_json(image_dir, output_path=None):
    """
    Create a temporary COCO JSON file for unlabeled images in a directory.
    
    Args:
        image_dir (str): Directory containing images
        output_path (str, optional): Where to save the temp JSON file.
                                   If None, creates it in a temp directory.
    
    Returns:
        str: Path to the created temporary COCO JSON file
    """
    # Get all image files
    image_files = [
        f for f in os.listdir(image_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
    ]
    
    # Create a minimal COCO dataset JSON structure
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add image info to the COCO dict
    for idx, image_file in enumerate(image_files):
        coco_dict["images"].append({
            "id": idx,
            "file_name": image_file,
            "width": 0,  # These will be updated by SAHI
            "height": 0  # These will be updated by SAHI
        })
    
    # Determine where to save the file
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        temp_coco_path = os.path.join(temp_dir, "temp_coco_dataset.json")
    else:
        temp_coco_path = output_path
    
    # Save the temporary COCO JSON
    with open(temp_coco_path, 'w') as f:
        json.dump(coco_dict, f)
    
    return temp_coco_path

def copy_coco_results(results_dict, output_dir, name):
    """
    Copy COCO annotation results to the desired output location
    
    Args:
        results_dict (dict): Results dictionary from SAHI predict function
        output_dir (str): Directory to save the final COCO annotations
        name (str): Name prefix for the output file
        
    Returns:
        str: Path to the final COCO annotations file
    """
    # Get the path to the generated COCO annotations
    result_json_path = Path(results_dict["export_dir"]) / "result.json"
    final_output_path = os.path.join(output_dir, f"{name}_coco_annotations.json")
    
    # Copy to final location
    shutil.copy(result_json_path, final_output_path)
    return final_output_path



def list_files_with_extensions(
    directory: Union[str, Path], 
    extensions: List[str], 
    recursive: bool = False
) -> List[str]:
    """
    Lists all files in a directory that match the specified extensions.
    
    Args:
        directory (str or Path): Directory to search for files
        extensions (List[str]): List of file extensions to filter by (e.g., ['.jpg', '.png'])
        recursive (bool, optional): Whether to search recursively in subdirectories. Defaults to False.
    
    Returns:
        List[str]: List of full paths to the matching files
    """
    # Normalize directory to Path object
    directory = Path(directory)
    
    # Make sure extensions have dots
    normalized_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    # Initialize result list
    matching_files = []
    
    # Walk through directory
    if recursive:
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = Path(root) / filename
                if file_path.suffix.lower() in normalized_extensions:
                    matching_files.append(str(file_path))
    else:
        # Non-recursive version
        for filename in os.listdir(directory):
            file_path = directory / filename
            if file_path.is_file() and file_path.suffix.lower() in normalized_extensions:
                matching_files.append(str(file_path))
    
    # Sort the files to ensure consistent order
    matching_files.sort()
    
    return matching_files