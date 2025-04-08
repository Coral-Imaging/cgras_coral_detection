#!/usr/bin/env python3

import os
import sys
import yaml
import time
import logging
import argparse
import random
import shutil
from pathlib import Path
from copy import deepcopy

# Import directly from the image_processing module what we can use
from image_processing import setup_logging, run_pipeline, load_config

class SubsetExperiment:
    """
    Class to run experiments with increasing subsets of images to evaluate
    model performance as training data size increases.
    """
    
    def __init__(self, config_path, subset_sizes=None, num_runs=1, seed=None):
        """
        Initialize the subset experiment.
        
        Args:
            config_path (str): Path to the base configuration YAML file.
            subset_sizes (list): List of subset sizes to use (e.g., [1, 10, 20, 30, ...]).
            num_runs (int): Number of times to run the experiment with different random selections.
            seed (int): Random seed for reproducibility.
        """
        self.logger = logging.getLogger("subset_experiment")
        self.config_path = config_path
        self.base_config = load_config(config_path, self.logger)
        
        # Set default subset sizes if not provided
        if subset_sizes is None:
            self.subset_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        else:
            self.subset_sizes = subset_sizes
            
        self.num_runs = num_runs
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            self.logger.info(f"Using random seed: {seed}")
        
        # Create temporary directory for subset datasets
        self.temp_dir = Path(self.base_config.get('output_base_path')) / "subset_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.config_dir = Path(self.base_config.get('output_base_path')) / "configs"
        os.makedirs(self.config_dir, exist_ok=True)
    
    def _load_dataset_info(self):
        """
        Load information about the full dataset from the YAML file.
        """
        input_path = self.base_config.get('input_path')
        if not input_path:
            self.logger.error("Input path not specified in config.")
            sys.exit(1)
            
        # Load the dataset YAML file
        try:
            with open(input_path, 'r') as f:
                dataset_yaml = yaml.safe_load(f)
                self.logger.info(f"Dataset loaded from {input_path}")
                return dataset_yaml
        except Exception as e:
            self.logger.error(f"Error loading dataset file: {e}")
            sys.exit(1)
    
    def _get_all_image_paths(self, dataset_yaml):
        """
        Extract all image paths from the dataset YAML.
        
        Args:
            dataset_yaml (dict): The loaded dataset YAML.
            
        Returns:
            list: List of all image paths.
        """
        image_paths = []
        
        # Get the base path from the YAML
        base_path = Path(dataset_yaml.get('path', '.'))
        
        # Get the data directory relative path(s)
        data_dirs = dataset_yaml.get('data', [])
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]
        
        # Collect all image files from these directories
        for data_dir in data_dirs:
            # If this is an images directory, look for image files
            if 'images' in Path(data_dir).parts:
                image_dir = base_path / data_dir
                self.logger.info(f"Looking for images in: {image_dir}")
                
                # If the directory exists, collect all image files
                if os.path.isdir(image_dir):
                    for root, _, files in os.walk(image_dir):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                                image_path = os.path.join(root, file)
                                # Store relative path from base_path
                                rel_path = os.path.relpath(image_path, base_path)
                                image_paths.append(rel_path)
                else:
                    self.logger.warning(f"Directory not found: {image_dir}")
        
        self.logger.info(f"Found {len(image_paths)} images in the dataset.")
        
        return image_paths
    
    def _create_subset_yaml(self, dataset_yaml, selected_images, output_path):
        """
        Create a new YAML file and copy the selected images to a new directory structure.
        
        Args:
            dataset_yaml (dict): The original dataset YAML.
            selected_images (list): List of image paths to include.
            output_path (str): Path to save the new YAML file.
            
        Returns:
            str: Path to the created YAML file.
        """
        # Create a deep copy of the dataset to modify
        subset_yaml = deepcopy(dataset_yaml)
        
        # Create a new unique base path for this subset
        # We'll create a subdirectory in the same parent directory as the output_path
        output_dir = Path(output_path).parent / f"subset_{Path(output_path).stem}"
        images_dir = output_dir / "data/images"
        labels_dir = output_dir / "data/labels"
        
        # Create the necessary directories
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Update the path in the YAML
        subset_yaml['path'] = str(output_dir)
        
        # Source base path
        source_base_path = Path(dataset_yaml.get('path', '.'))
        
        # Copy the selected images and their corresponding labels
        copied_count = 0
        for rel_image_path in selected_images:
            # Source paths
            source_image_path = source_base_path / rel_image_path
            
            # Create label path by replacing 'images' with 'labels' and changing extension to .txt
            rel_label_path = str(rel_image_path).replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            source_label_path = source_base_path / rel_label_path
            
            # Destination paths - keep the same file structure
            # Extract just the filename to flatten directory structure
            image_filename = Path(rel_image_path).name
            label_filename = Path(rel_label_path).name
            
            dest_image_path = images_dir / image_filename
            dest_label_path = labels_dir / label_filename
            
            # Copy image file
            if os.path.exists(source_image_path):
                shutil.copy2(source_image_path, dest_image_path)
                copied_count += 1
            else:
                self.logger.warning(f"Image not found: {source_image_path}")
            
            # Copy label file if it exists
            if os.path.exists(source_label_path):
                shutil.copy2(source_label_path, dest_label_path)
            else:
                self.logger.warning(f"Label not found: {source_label_path}")
        
        # Update the data entry to point to the new flattened structure
        subset_yaml['data'] = ['data/images']
        
        # Save the new YAML file
        new_yaml_path = output_dir / "cgras_data.yaml"
        with open(new_yaml_path, 'w') as f:
            yaml.dump(subset_yaml, f, default_flow_style=False)
            
        self.logger.info(f"Created subset YAML with {copied_count} images at {new_yaml_path}")
        self.logger.info(f"Subset data directory: {output_dir}")
        
        return str(new_yaml_path)
    
    def run_experiment(self):
        """
        Run the subset experiment for all specified subset sizes and runs.
        """
        # Load dataset information
        dataset_yaml = self._load_dataset_info()
        all_images = self._get_all_image_paths(dataset_yaml)
        
        # Ensure we have enough images for the largest subset
        max_subset_size = max(self.subset_sizes)
        if len(all_images) < max_subset_size:
            self.logger.error(f"Not enough images in dataset. Need {max_subset_size}, found {len(all_images)}.")
            sys.exit(1)
        
        # Run the experiment for each specified number of runs
        for run_idx in range(1, self.num_runs + 1):
            self.logger.info(f"Starting run {run_idx} of {self.num_runs}")
            
            # Shuffle all images for this run
            random.shuffle(all_images)
            
            # Use the first N images for all subsets (cumulative approach)
            cumulative_images = []
            
            # For each subset size, create and process the subset
            for subset_size in self.subset_sizes:
                self.logger.info(f"Processing subset of {subset_size} images")
                
                # Add more images to reach the current subset size
                while len(cumulative_images) < subset_size:
                    next_image = all_images[len(cumulative_images)]
                    cumulative_images.append(next_image)
                
                # Create current subset (which includes all previous images)
                current_subset = cumulative_images[:subset_size]
                
                # Create a unique name for this subset
                subset_name = f"run{run_idx}_{subset_size}"
                
                # Create a temporary YAML file for this subset
                subset_yaml_path = self.temp_dir / f"{subset_name}.yaml"
                new_yaml_path = self._create_subset_yaml(dataset_yaml, current_subset, subset_yaml_path)

                # Create a modified config for this subset
                subset_config = deepcopy(self.base_config)
                subset_config['input_path'] = str(new_yaml_path)
                subset_config['project_name'] = f"{subset_config.get('project_name', 'default')}_{subset_name}"
                
                # Save the modified config
                subset_config_path = self.config_dir / f"config_{subset_name}.yaml"
                with open(subset_config_path, 'w') as f:
                    yaml.dump(subset_config, f, default_flow_style=False)
                
                # Run the pipeline with this subset
                self.logger.info(f"Running pipeline for {subset_name}")
                run_pipeline(subset_config, self.logger)
        
        self.logger.info("All subset experiments completed successfully!")
        
        # Clean up temporary files if needed
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.config_dir)
        self.logger.info("Temporary files cleaned up.")

def main():
    """Main entry point for the subset experiment script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run subset experiments for image processing pipeline')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to the base configuration YAML file')
    parser.add_argument('--sizes', '-s', nargs='+', type=int,
                        default=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
                        help='List of subset sizes to use (e.g., 1 10 20 30 ...)')
    parser.add_argument('--runs', '-r', type=int, default=1,
                        help='Number of times to run the experiment with different random selections')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)
    
    # Record start time
    start_time = time.time()
    
    # Run the experiment
    try:
        experiment = SubsetExperiment(
            args.config, 
            subset_sizes=args.sizes,
            num_runs=args.runs,
            seed=args.seed
        )
        experiment.run_experiment()
    except Exception as e:
        logger.exception(f"Experiment failed with error: {e}")
        sys.exit(1)
    
    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    logger.info(f"Experiment completed in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()