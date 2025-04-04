#!/usr/bin/env python3

import os
import sys
import yaml
import time
import logging
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import processing modules
from utils.folder_structurer import FolderStructurer
from utils.filterer import ImageFilterer
from utils.data_splitter import DatasetSplitter
from utils.image_patcher import ImagePatcher
from utils.balancer import DatasetBalancer

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log', mode='w')
        ]
    )
    return logging.getLogger("pipeline")

def load_config(config_path):
    """Load the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)

def run_folder_structure(config, input_path, output_path):
    """Run the folder structure step."""
    logger.info("Starting folder structure step...")
    
    structurer = FolderStructurer(input_path, output_path)
    
    if structurer.validate_input():
        success = structurer.convert()
        if success:
            logger.info(f"Folder structure step completed. Output at: {output_path}")
            return structurer.yaml_path
        else:
            logger.error("Folder structure step failed during conversion.")
            sys.exit(1)
    else:
        logger.error("Folder structure validation failed.")
        sys.exit(1)

def run_filter(config, input_yaml_path, output_path):
    """Run the filter step."""
    logger.info("Starting filter step...")
    
    filter_config = config.get('filter', {})
    
    # Parse min_pixel_area and class_ids
    min_pixel_area_list = filter_config.get('min_pixel_area', [175, 175, 200, 200])
    class_ids = filter_config.get('class_ids', [0, 1, 2, 3])    
    copy_images = filter_config.get('copy_images', True)
    
    # Initialize the filterer
    filterer = ImageFilterer(input_yaml_path, output_path)
    
    # Analyze before filtering (but don't display)
    filterer.analyze_dataset_areas()
    
    # Run filtering
    filterer.filter_small_labels(
        min_pixel_area=min_pixel_area_list,
        class_ids=class_ids,
        copy_images=copy_images
    )
    
    logger.info(f"Filter step completed. Output at: {output_path}")
    return filterer.new_yaml_path

def run_split(config, input_yaml_path, output_path):
    """Run the split step."""
    logger.info("Starting split step...")
    
    split_config = config.get('split', {})
    
    # Get split parameters
    split_field = split_config.get('split_field', 'tile')
    train_ratio = split_config.get('train_ratio', 0.7)
    val_ratio = split_config.get('val_ratio', 0.15)
    test_ratio = split_config.get('test_ratio', 0.15)
    stratify_by = split_config.get('stratify_by')
    random_seed = split_config.get('random_seed', 42)
    
    # Initialize the splitter
    splitter = DatasetSplitter(input_yaml_path, output_path)
    
    # Parse file info
    splitter.parse_file_info()
    
    # Create splits
    _ = splitter.create_splits(
        split_field=split_field,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by=stratify_by,
        random_seed=random_seed
    )
    
    # Export splits
    success = splitter.export_splits()
    
    if success:
        logger.info(f"Split step completed. Output at: {output_path}")
        return splitter.new_yaml_path
    else:
        logger.error("Split step failed during export.")
        sys.exit(1)

def run_patch(config, input_yaml_path, output_path):
    """Run the patch (tiling) step."""
    logger.info("Starting patch step...")
    
    patch_config = config.get('patch', {})
    
    # Get patch parameters
    tile_width = patch_config.get('tile_width', 640)
    tile_height = patch_config.get('tile_height', 640)
    truncate_percent = patch_config.get('truncate_percent', 0.5)
    max_files = patch_config.get('max_files', 16382)
    
    # Initialize the patcher
    patcher = ImagePatcher(
        input_yaml_path, 
        output_path,
        tile_width=tile_width,
        tile_height=tile_height,
        truncate_percent=truncate_percent,
        max_files=max_files,
    )
    
    # Process datasets
    patcher.process_all_datasets()
    
    logger.info(f"Patch step completed. Output at: {output_path}")
    return patcher.new_yaml_path

def run_balance(config, input_yaml_path, output_path):
    """Run the balance step."""
    logger.info("Starting balance step...")
      
    # Initialize the balancer
    balancer = DatasetBalancer(input_yaml_path, output_path)
    
    # Balance datasets
    _ = balancer.balance_datasets()
    
    logger.info(f"Balance step completed. Output at: {output_path}")
    return balancer.new_yaml_path

def run_pipeline(config):
    """Run the entire pipeline based on the configuration."""
    pipeline_steps = config.get('pipeline', [])
    project_name = config.get('project_name', 'default')
    input_path = config.get('input_path')
    output_base_path = config.get('output_base_path')
    
    if not input_path or not output_base_path:
        logger.error("Input and output paths must be specified in the config.")
        sys.exit(1)
    
    # Convert to Path objects
    input_path = Path(input_path)
    output_base_path = Path(output_base_path)
    
    # Initialize output path with just the project name
    current_output_path = output_base_path / project_name
    current_yaml_path = None
    
    # Create a chain of output paths based on the enabled steps
    step_suffix_map = {
        'folder_structure': '',  # No suffix for the first step
        'filter': 'filtered',
        'split': 'split',
        'patch': 'tiled',
        'balance': 'balanced'
    }
    
    # Track steps run and build output path
    output_suffix = ''
    
    # Run each step in the pipeline
    for step in pipeline_steps:
        step_config = config.get(step, {})
        enabled = step_config.get('enabled', True)
        
        if not enabled:
            logger.info(f"Skipping {step} step (disabled in config)")
            continue
        
        # Update output path suffix
        if step_suffix_map[step]:
            if output_suffix:
                output_suffix += f"_{step_suffix_map[step]}"
            else:
                output_suffix = step_suffix_map[step]
        
        # Determine current output path
        if output_suffix:
            current_output_path = output_base_path / f"{project_name}_{output_suffix}"
        else:
            current_output_path = output_base_path / project_name
        
        # Make sure output directory exists
        os.makedirs(current_output_path, exist_ok=True)
        
        # Run the appropriate step
        logger.info(f"Running step: {step} with output to {current_output_path}")
        
        if step == 'folder_structure':
            current_yaml_path = run_folder_structure(config, input_path, current_output_path)
        elif step == 'filter':
            if not current_yaml_path:
                logger.error("No input YAML path available for filter step.")
                sys.exit(1)
            current_yaml_path = run_filter(config, current_yaml_path, current_output_path)
        elif step == 'split':
            if not current_yaml_path:
                logger.error("No input YAML path available for split step.")
                sys.exit(1)
            current_yaml_path = run_split(config, current_yaml_path, current_output_path)
        elif step == 'patch':
            if not current_yaml_path:
                logger.error("No input YAML path available for patch step.")
                sys.exit(1)
            current_yaml_path = run_patch(config, current_yaml_path, current_output_path)
        elif step == 'balance':
            if not current_yaml_path:
                logger.error("No input YAML path available for balance step.")
                sys.exit(1)
            current_yaml_path = run_balance(config, current_yaml_path, current_output_path)
        else:
            logger.warning(f"Unknown pipeline step: {step}, skipping.")

    logger.info("Pipeline completed successfully!")
    logger.info(f"Final output path: {current_output_path}")
    logger.info(f"Final YAML path: {current_yaml_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run CGRAS data processing pipeline')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)
    
    # Record start time
    start_time = time.time()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run the pipeline
    try:
        run_pipeline(config)
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)
    
    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    logger.info(f"Pipeline completed in {duration:.2f} seconds.")