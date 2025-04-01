#!/usr/bin/env python3

"""
Labeling.py
A complete pipeline for processing and preparing image datasets for YOLO training:
1. Splits dataset into train/test/validation sets
2. Generates image patches from the split datasets
3. Balances the dataset by equalizing empty and non-empty labels

This pipeline uses temporary storage between steps to avoid permanent intermediate files.
Configuration can be loaded from a YAML file.
"""

import os
import argparse
import tempfile
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
import yaml

# Import our pipeline components
from splitfiles import DatasetSplitter
from ImagePatcher import ImagePatcher
from DatasetBalancer import DatasetBalancer


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return {}


class LabelingPipeline:
    """
    Orchestrates the complete labeling pipeline process from raw data to 
    training-ready balanced dataset with patches.
    """

    def __init__(self, 
                 input_dir, 
                 output_dir,
                 train_ratio=0.70,
                 test_ratio=0.15,
                 valid_ratio=0.15,
                 tile_width=640,
                 tile_height=640,
                 truncate_percent=0.5,
                 empty_threshold=10,
                 max_files=16382,
                 num_workers=None,
                 keep_temp=False,
                 log_level='INFO'):
        """
        Initialize the labeling pipeline with configuration parameters.
        
        Args:
            input_dir: Directory containing the raw dataset (must have images and labels subdirs)
            output_dir: Directory where the final processed dataset will be saved
            train_ratio: Portion of data used for training (default: 0.70)
            test_ratio: Portion of data used for testing (default: 0.15)
            valid_ratio: Portion of data used for validation (default: 0.15)
            tile_width: Width of image patches in pixels (default: 640)
            tile_height: Height of image patches in pixels (default: 640)
            truncate_percent: Minimum percentage of an object that must be in a tile (default: 0.5)
            empty_threshold: Size in bytes below which a label file is considered empty (default: 10)
            max_files: Maximum files per directory (default: 16382)
            num_workers: Number of worker processes (default: None, uses system CPU count)
            keep_temp: Whether to keep temporary files (default: False)
            log_level: Logging level (default: 'INFO')
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.truncate_percent = truncate_percent
        self.empty_threshold = empty_threshold
        self.max_files = max_files
        self.num_workers = num_workers
        self.keep_temp = keep_temp
        
        # Setup logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'labeling_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger('LabelingPipeline')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate input directory
        if not os.path.exists(os.path.join(input_dir, 'images')) or not os.path.exists(os.path.join(input_dir, 'labels')):
            raise ValueError(f"Input directory {input_dir} must contain 'images' and 'labels' subdirectories")
    
    def create_temp_dir(self):
        """Create a temporary directory for intermediate processing steps."""
        temp_dir = tempfile.mkdtemp(prefix="labeling_pipeline_")
        self.logger.info(f"Created temporary directory: {temp_dir}")
        return temp_dir
    
    def cleanup_temp_dir(self, temp_dir):
        """Clean up a temporary directory if keep_temp is False."""
        if not self.keep_temp:
            self.logger.info(f"Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            self.logger.info(f"Keeping temporary directory: {temp_dir}")
    
    def step1_split_dataset(self, temp_dir):
        """
        Step 1: Split the dataset into train/test/validation sets.
        
        Args:
            temp_dir: Temporary directory for storing the split datasets
            
        Returns:
            Dictionary with paths to the split datasets
        """
        self.logger.info("Step 1: Splitting dataset into train/test/validation sets")
        start_time = time.time()
        
        # Create a DatasetSplitter instance
        splitter = DatasetSplitter(
            data_location=self.input_dir,
            save_dir=temp_dir,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            valid_ratio=self.valid_ratio,
            max_files=self.max_files
        )
        
        # Split the dataset
        split_results = splitter.split_dataset()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Dataset split completed in {elapsed_time:.2f} seconds")
        
        # Return the paths to the split datasets
        return {
            'train': os.path.join(temp_dir, 'train'),
            'valid': os.path.join(temp_dir, 'valid'),
            'test': os.path.join(temp_dir, 'test')
        }
    
    def step2_generate_patches(self, split_paths, temp_dir):
        """
        Step 2: Generate patches from the split datasets.
        
        Args:
            split_paths: Dictionary with paths to the split datasets
            temp_dir: Temporary directory for storing the patched datasets
            
        Returns:
            Dictionary with paths to the patched datasets
        """
        self.logger.info("Step 2: Generating patches from split datasets")
        start_time = time.time()
        
        patch_paths = {}
        
        # Create output directories for patches
        patches_dir = os.path.join(temp_dir, 'patches')
        os.makedirs(patches_dir, exist_ok=True)
        
        # Process each split
        for split_name, split_path in split_paths.items():
            self.logger.info(f"Processing {split_name} split...")
            
            # Find all subdirectories that might have been created due to max_files limit
            split_dirs = []
            if os.path.exists(split_path):
                # Check if this is a simple split directory
                if os.path.exists(os.path.join(split_path, 'images')) and os.path.exists(os.path.join(split_path, 'labels')):
                    split_dirs.append(split_path)
                else:
                    # Check for numbered subdirectories
                    for item in os.listdir(split_path):
                        item_path = os.path.join(split_path, item)
                        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'images')) and os.path.exists(os.path.join(item_path, 'labels')):
                            split_dirs.append(item_path)
            
            # If no valid directories found, log a warning and continue
            if not split_dirs:
                self.logger.warning(f"No valid data directories found in {split_path}")
                continue
            
            # Process each sub-directory
            for i, dir_path in enumerate(split_dirs):
                output_subdir = os.path.join(patches_dir, f"{split_name}_{i}")
                os.makedirs(output_subdir, exist_ok=True)
                
                # Create patcher
                patcher = ImagePatcher(
                    full_res_dir=dir_path,
                    save_path=output_subdir,
                    tile_width=self.tile_width,
                    tile_height=self.tile_height,
                    truncate_percent=self.truncate_percent,
                    max_files=self.max_files,
                    num_workers=self.num_workers
                )
                
                # Process images
                processed_count, total_tiles = patcher.process_image_list()
                self.logger.info(f"Processed {processed_count} images with {total_tiles} tiles from {dir_path}")
            
            # Store the path to the patched dataset
            patch_paths[split_name] = os.path.join(patches_dir, split_name)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Patch generation completed in {elapsed_time:.2f} seconds")
        
        return patch_paths
    
    def step3_balance_dataset(self, patch_paths):
        """
        Step 3: Balance the dataset by equalizing empty and non-empty labels.
        
        Args:
            patch_paths: Dictionary with paths to the patched datasets
            
        Returns:
            True if successful
        """
        self.logger.info("Step 3: Balancing dataset")
        start_time = time.time()
        
        # Create a DatasetBalancer instance
        balancer = DatasetBalancer(
            root_folder=os.path.dirname(list(patch_paths.values())[0]),  # Use the parent directory of the first path
            destination_folder=self.output_dir,
            empty_threshold=self.empty_threshold,
            max_workers=self.num_workers
        )
        
        # Balance the dataset
        total_processed, total_missing = balancer.balance_dataset()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Dataset balancing completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Processed {total_processed} files. Missing images: {total_missing}")
        
        return True
    
    def run_pipeline(self):
        """Run the complete labeling pipeline."""
        self.logger.info(f"Starting labeling pipeline: {self.input_dir} -> {self.output_dir}")
        start_time = time.time()
        
        # Create a temporary directory for intermediate results
        temp_dir = self.create_temp_dir()
        
        try:
            # Step 1: Split the dataset
            split_paths = self.step1_split_dataset(temp_dir)
            
            # Step 2: Generate patches
            patch_paths = self.step2_generate_patches(split_paths, temp_dir)
            
            # Step 3: Balance the dataset
            self.step3_balance_dataset(patch_paths)
            
            # Pipeline complete
            elapsed_time = time.time() - start_time
            self.logger.info(f"Labeling pipeline completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Final dataset available at: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up temporary directory
            self.cleanup_temp_dir(temp_dir)


def main():
    """Command line entry point for the labeling pipeline."""
    parser = argparse.ArgumentParser(description='Run the complete labeling pipeline')
    
    # Config file
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config.yaml'),
                       help='Path to YAML configuration file (default: config.yaml in script directory)')
    
    # Input and output
    parser.add_argument('--input', type=str,
                       help='Input directory containing images and labels subdirectories')
    parser.add_argument('--output', type=str,
                       help='Output directory for the final processed dataset')
    
    # Dataset splitting parameters
    parser.add_argument('--train-ratio', type=float,
                       help='Portion of data used for training (default: 0.70)')
    parser.add_argument('--test-ratio', type=float,
                       help='Portion of data used for testing (default: 0.15)')
    parser.add_argument('--valid-ratio', type=float,
                       help='Portion of data used for validation (default: 0.15)')
    
    # Image patching parameters
    parser.add_argument('--tile-width', type=int,
                       help='Width of image patches in pixels (default: 640)')
    parser.add_argument('--tile-height', type=int,
                       help='Height of image patches in pixels (default: 640)')
    parser.add_argument('--truncate', type=float,
                       help='Minimum percentage of an object that must be in a tile (default: 0.5)')
    
    # Dataset balancing parameters
    parser.add_argument('--empty-threshold', type=int,
                       help='Size in bytes below which a label file is considered empty (default: 10)')
    
    # General parameters
    parser.add_argument('--max-files', type=int,
                       help='Maximum files per directory (default: 16382)')
    parser.add_argument('--workers', type=int,
                       help='Number of worker processes (default: None, uses system CPU count)')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary files for debugging (default: False)')
    parser.add_argument('--log-level', type=str,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Merge command line arguments with config file, giving priority to command line
    input_dir = args.input or config.get('data_location')
    output_dir = args.output or config.get('final_output_dir')
    train_ratio = args.train_ratio or config.get('train_ratio', 0.70)
    test_ratio = args.test_ratio or config.get('test_ratio', 0.15)
    valid_ratio = args.valid_ratio or config.get('valid_ratio', 0.15)
    tile_width = args.tile_width or config.get('tile_width', 640)
    tile_height = args.tile_height or config.get('tile_height', 640)
    truncate = args.truncate or config.get('truncate_percent', 0.5)
    max_files = args.max_files or config.get('max_files_per_dir', 16382)
    workers = args.workers if args.workers is not None else config.get('workers')
    empty_threshold = args.empty_threshold or 10
    log_level = args.log_level or 'INFO'
    
    # Make sure required parameters are provided
    if not input_dir:
        parser.error("Input directory is required (provide via --input or config file)")
    if not output_dir:
        parser.error("Output directory is required (provide via --output or config file)")
    
    # Validate ratios
    total_ratio = train_ratio + test_ratio + valid_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"The sum of train, test, and valid ratios must be 1.0 (got {total_ratio})")
    
    # Create and run the pipeline
    pipeline = LabelingPipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        tile_width=tile_width,
        tile_height=tile_height,
        truncate_percent=truncate,
        empty_threshold=empty_threshold,
        max_files=max_files,
        num_workers=workers,
        keep_temp=args.keep_temp,
        log_level=log_level
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
