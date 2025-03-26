#! /usr/bin/env python3

""" process_dataset.py
    This script orchestrates the processing of CVAT annotated datasets by:
    1. Balancing empty/non-empty images using NegRemover
    2. Splitting the balanced dataset into train/val/test using DatasetSplitter
    3. Tiling the images within each split using ImageTiler
"""

import os
import sys
import yaml
import glob
from typing import Dict, Any

# Add parent directory to path to allow importing from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.image_tiler import ImageTiler
from utils.file_splitter import DatasetSplitter
from utils.neg_remover import NegRemover

# Configuration from environment variables or defaults
def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables or use defaults."""
    return {
        "DATA_PATH": os.environ.get("DATA_PATH", "/media/java/RRAP03"),
        "INPUT_FOLDER": os.environ.get("INPUT_FOLDER", "export100_from_cvat"),
        "OUTPUT_FOLDER": os.environ.get("OUTPUT_FOLDER", "processed_dataset"),
        "TRAIN_SPLIT": float(os.environ.get("TRAIN_SPLIT", "0.7")),
        "VAL_SPLIT": float(os.environ.get("VAL_SPLIT", "0.15")),
        "TEST_SPLIT": float(os.environ.get("TEST_SPLIT", "0.15")),
        "TILE_WIDTH": int(os.environ.get("TILE_WIDTH", "640")),
        "TILE_HEIGHT": int(os.environ.get("TILE_HEIGHT", "640")),
        "OVERLAP": int(os.environ.get("OVERLAP", "50")),
        "CLASSES": None,  # None for all classes. Can select classes => [0, 1, 2, 3]
        "VERBOSE": os.environ.get("VERBOSE", "True").lower() == "true"
    }

def setup_directories(config: Dict[str, Any]) -> Dict[str, str]:
    """Create the necessary directory structure."""
    base_input_dir = os.path.join(config["DATA_PATH"], config["INPUT_FOLDER"])
    outputs_dir = os.path.join(config["DATA_PATH"], config["OUTPUT_FOLDER"])
    
    # Create directories for each processing stage
    balanced_output_dir = os.path.join(outputs_dir, "balanced_dataset")
    split_output_dir = os.path.join(outputs_dir, "split_dataset")
    final_output_dir = os.path.join(outputs_dir, "final_dataset")
    yaml_output_path = os.path.join(outputs_dir, "cgras_data.yaml")
    
    # Create directories
    for directory in [outputs_dir, balanced_output_dir, split_output_dir, final_output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        "base_input": base_input_dir,
        "outputs": outputs_dir,
        "balanced": balanced_output_dir,
        "split": split_output_dir,
        "final": final_output_dir,
        "yaml_path": yaml_output_path
    }

def balance_dataset(input_path: str, output_path: str, dataset_name: str, verbose: bool) -> Dict[str, Any]:
    """Balance the dataset by removing excess negative samples."""
    if verbose:
        print(f"\nBalancing dataset: {dataset_name}")
    
    dataset_output_path = os.path.join(output_path, dataset_name)
    os.makedirs(dataset_output_path, exist_ok=True)
    
    remover = NegRemover(
        root_folder=input_path,
        destination_folder=dataset_output_path
    )
    
    stats = remover.process_all(verbose=verbose)
    
    if verbose:
        print(f"Balancing completed for {dataset_name}")
    
    return stats

def split_dataset(input_path: str, output_path: str, dataset_name: str, config: Dict[str, Any], verbose: bool) -> str:
    """Split the dataset into train/val/test sets."""
    if verbose:
        print(f"\nSplitting dataset: {dataset_name}")
    
    dataset_output_path = os.path.join(output_path, dataset_name)
    os.makedirs(dataset_output_path, exist_ok=True)
    
    splitter = DatasetSplitter(
        data_location=input_path,
        save_dir=dataset_output_path,
        train_ratio=config["TRAIN_SPLIT"],
        valid_ratio=config["VAL_SPLIT"],
        test_ratio=config["TEST_SPLIT"]
    )
    
    stats = splitter.split_dataset()
    
    if verbose:
        print(f"Splitting completed for {dataset_name}")
        print(f"  - Training: {stats['train']} files")
        print(f"  - Validation: {stats['valid']} files")
        print(f"  - Test: {stats['test']} files")
    
    return dataset_output_path

def tile_images_in_split(input_path: str, output_path: str, split_type: str, dataset_name: str, config: Dict[str, Any], verbose: bool) -> None:
    """Tile images in a single split (train/val/test)."""
    if verbose:
        print(f"\nTiling {split_type} images for {dataset_name}")
    
    # Handle the case where train might be split into multiple dirs (train_0, train_1, etc.)
    split_dirs = []
    if split_type == "train":
        # Check for numbered train directories (train_0, train_1, etc.)
        train_dirs = glob.glob(os.path.join(input_path, f"{split_type}_*"))
        if train_dirs:
            split_dirs.extend(train_dirs)
        else:
            split_dirs.append(os.path.join(input_path, split_type))
    else:
        split_dirs.append(os.path.join(input_path, split_type))
    
    for split_dir in split_dirs:
        if not os.path.exists(split_dir):
            if verbose:
                print(f"Warning: Split directory {split_dir} does not exist. Skipping.")
            continue
            
        split_name = os.path.basename(split_dir)
        output_split_path = os.path.join(output_path, dataset_name, split_name)
        os.makedirs(output_split_path, exist_ok=True)
        
        # Create a data.yaml file for the tiler to use
        with open(os.path.join(split_dir, "data.yaml"), "w") as f:
            yaml.dump({
                "names": {0: "alive", 1: "dead", 2: "mask_live", 3: "mask_dead"},
                "path": ".",
                "Train": "Train.txt"
            }, f)
        
        # Create a Train.txt file with paths to all images
        image_files = glob.glob(os.path.join(split_dir, "images", "*.jpg"))
        with open(os.path.join(split_dir, "Train.txt"), "w") as f:
            for img_file in image_files:
                rel_path = os.path.join("images", os.path.basename(img_file))
                f.write(f"{rel_path}\n")
        
        # Tile images - Fix the repeated data_path parameter
        tiler = ImageTiler(
            data_path=split_dir,
            output_path=output_split_path,
            tile_size=(config["TILE_WIDTH"], config["TILE_HEIGHT"]),
            overlap_percent=config["OVERLAP"],
            wanted_classes=config["CLASSES"]
        )
        
        tiler.tile_images()
        
        if verbose:
            print(f"Tiling completed for {dataset_name}/{split_name}")

def update_yaml_file(dirs: Dict[str, str], verbose: bool) -> None:
    """Update the cgras_data.yaml file with paths to all datasets."""
    dataset_dirs = {}
    
    # Find all train, val, test directories in the final output
    for dataset_name in os.listdir(dirs["final"]):
        dataset_path = os.path.join(dirs["final"], dataset_name)
        if os.path.isdir(dataset_path):
            train_dirs = []
            val_dirs = []
            test_dirs = []
            
            # Find train directories (including train_0, train_1, etc.)
            for train_dir in glob.glob(os.path.join(dataset_path, "train*")):
                if os.path.isdir(os.path.join(train_dir, "images")):
                    rel_path = os.path.relpath(train_dir, dirs["final"])
                    train_dirs.append(f"{rel_path}/images")
            
            # Find val directory
            val_dir = os.path.join(dataset_path, "valid")
            if os.path.isdir(os.path.join(val_dir, "images")):
                rel_path = os.path.relpath(val_dir, dirs["final"])
                val_dirs.append(f"{rel_path}/images")
            
            # Find test directory
            test_dir = os.path.join(dataset_path, "test")
            if os.path.isdir(os.path.join(test_dir, "images")):
                rel_path = os.path.relpath(test_dir, dirs["final"])
                test_dirs.append(f"{rel_path}/images")
            
            dataset_dirs[dataset_name] = {
                "train": train_dirs,
                "val": val_dirs,
                "test": test_dirs
            }
    
    # Create combined lists for all datasets
    all_train = []
    all_val = []
    all_test = []
    
    for dataset in dataset_dirs.values():
        all_train.extend(dataset["train"])
        all_val.extend(dataset["val"])
        all_test.extend(dataset["test"])
    
    # Create the YAML data
    yaml_data = {
        "path": dirs["final"],
        "train": all_train,
        "val": all_val,
        "test": all_test,
        "names": {0: "alive", 1: "dead", 2: "mask_live", 3: "mask_dead"}
    }
    
    # Write the YAML file
    with open(dirs["yaml_path"], "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    
    if verbose:
        print(f"\nDataset processing complete. The dataset details have been saved to {dirs['yaml_path']}.")
        print("\nTraining Configuration:")
        print(f"   - Dataset path: {dirs['final']}")
        print(f"   - Train images: {len(all_train)} directories")
        print(f"   - Validation images: {len(all_val)} directories")
        print(f"   - Test images: {len(all_test)} directories")

def main():
    """Main function to process datasets."""
    # Get configuration
    config = get_config()
    verbose = config["VERBOSE"]
    
    if verbose:
        print("Starting dataset processing with configuration:")
        for key, value in config.items():
            print(f"  - {key}: {value}")
    
    # Setup directories
    dirs = setup_directories(config)
    
    # Process each dataset folder
    dataset_folders = sorted(glob.glob(os.path.join(dirs["base_input"], "*")))
    
    if not dataset_folders:
        print(f"No dataset folders found in {dirs['base_input']}. Exiting.")
        return
    
    for dataset_path in dataset_folders:
        dataset_name = os.path.basename(dataset_path)
        if verbose:
            print(f"\nProcessing dataset: {dataset_name}")
        
        # # Step 1: Balance dataset (remove excess negative samples)
        # balance_dataset(dataset_path, dirs["balanced"], dataset_name, verbose)
        # input_for_split = os.path.join(dirs["balanced"], dataset_name)
        
        input_for_split = dataset_path

        # Step 2: Split dataset into train/val/test
        split_dataset_path = split_dataset(input_for_split, dirs["split"], dataset_name, config, verbose)
        
        # Step 3: Tile images within each split
        for split_type in ["train", "valid", "test"]:
            tile_images_in_split(split_dataset_path, dirs["final"], split_type, dataset_name, config, verbose)
    
    # Step 4: Update the YAML file
    update_yaml_file(dirs, verbose)

if __name__ == "__main__":
    main()