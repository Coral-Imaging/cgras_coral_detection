#!/usr/bin/env python3

"""
image_selector.py
Selects incremental subsets of images for training, ensuring each larger set contains all images
from smaller sets. Supports both linear and logarithmic scaling of image counts.
"""

import os
import re
import yaml
import random
import argparse
import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict


class ImageSelector:
    """Selects image subsets for incremental model training experiments."""
    
    def __init__(self, data_yaml_path, output_base_path, experiment_name="CoralScaling"):
        """
        Initialize the image selector.
        
        Args:
            data_yaml_path: Path to the cgras_data.yaml file
            output_base_path: Base path for outputs
            experiment_name: Name of the experiment
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.output_base_path = Path(output_base_path)
        self.experiment_name = experiment_name
        self.image_list = []
        self.data_config = None
        self.base_path = None
        self.image_metadata = defaultdict(list)  # For stratified sampling
        
        # Load YAML and validate paths
        self._load_data_yaml()
        
    def _load_data_yaml(self):
        """Load and validate the YAML configuration file."""
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.data_yaml_path}")
        
        with open(self.data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.base_path = Path(self.data_config['path'])
        
        # Validate image directory
        image_path = self.base_path / self.data_config['data'][0]
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_path}")
            
        # Get all images
        self.image_list = sorted([f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if not self.image_list:
            raise ValueError(f"No images found in {image_path}")
            
        print(f"Found {len(self.image_list)} images in {image_path}")
        
        # Parse image metadata for stratified sampling
        self._parse_image_metadata()
        
    def _parse_image_metadata(self):
        """Parse metadata from image filenames for stratified sampling."""
        pattern = r'CGRAS_(?P<species>\w+)_\w+_\d+_w(?P<week>\d+)_T(?P<tile>\d+)_(?P<index>\d+)'
        
        for img in self.image_list:
            match = re.match(pattern, img)
            if match:
                metadata = match.groupdict()
                # Store images by week number for stratified sampling
                week = int(metadata['week'])
                self.image_metadata['week'][week].append(img)
                # Store by tile number
                tile = int(metadata['tile'])
                self.image_metadata['tile'][tile].append(img)
                
        print(f"Parsed metadata for {sum(len(imgs) for imgs in self.image_metadata['week'].values())} images")
        print(f"Images span {len(self.image_metadata['week'])} different weeks")
        print(f"Images span {len(self.image_metadata['tile'])} different tiles")
        
    def generate_image_counts(self, num_sets=10, max_images=None, scaling='log'):
        """
        Generate the number of images for each training set.
        
        Args:
            num_sets: Number of different sized sets to create
            max_images: Maximum number of images to use (defaults to all available)
            scaling: 'log' or 'linear' scaling of image counts
            
        Returns:
            List of image counts for each set
        """
        if max_images is None:
            max_images = len(self.image_list)
        else:
            max_images = min(max_images, len(self.image_list))
            
        if scaling == 'log':
            # Logarithmic scaling (1, 2, 4, 8, 16, etc.)
            counts = np.logspace(0, np.log10(max_images), num=num_sets, dtype=int)
            # Ensure uniqueness and proper ordering
            counts = sorted(list(set(counts)))
            # Ensure we include the max count
            if counts[-1] < max_images:
                counts[-1] = max_images
        else:
            # Linear scaling
            counts = np.linspace(1, max_images, num=num_sets, dtype=int)
            
        # Handle any duplicates from rounding
        counts = sorted(list(set(counts)))
        
        return counts
        
    def select_images(self, num_images, seed=None, strategy='random'):
        """
        Select a specific number of images using the specified strategy.
        
        Args:
            num_images: Number of images to select
            seed: Random seed for reproducibility
            strategy: 'random', 'stratified_week', or 'stratified_tile'
            
        Returns:
            List of selected image filenames
        """
        if seed is not None:
            random.seed(seed)
            
        if num_images >= len(self.image_list):
            return self.image_list.copy()
            
        if strategy == 'random':
            return random.sample(self.image_list, num_images)
            
        elif strategy.startswith('stratified_'):
            strat_key = strategy.split('_')[1]  # 'week' or 'tile'
            selected = []
            
            # Calculate how many to take from each category
            categories = list(self.image_metadata[strat_key].keys())
            images_per_category = num_images / len(categories)
            
            # Distribute images across categories
            for category in categories:
                category_images = self.image_metadata[strat_key][category]
                # Calculate how many to take from this category
                n_from_category = min(len(category_images), 
                                     int(images_per_category) + (1 if random.random() < images_per_category % 1 else 0))
                
                selected.extend(random.sample(category_images, n_from_category))
                
                if len(selected) >= num_images:
                    break
                    
            # If we don't have enough (due to rounding), add more randomly
            if len(selected) < num_images:
                remaining = set(self.image_list) - set(selected)
                selected.extend(random.sample(list(remaining), num_images - len(selected)))
                
            return selected[:num_images]  # Ensure we have exactly num_images
        
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
            
    def create_incremental_sets(self, num_sets=10, max_images=None, scaling='log', 
                              seed=None, strategy='random'):
        """
        Create incremental sets of images where each larger set contains all images from smaller sets.
        
        Args:
            num_sets: Number of different sized sets to create
            max_images: Maximum number of images to use
            scaling: 'log' or 'linear' scaling
            seed: Random seed for reproducibility
            strategy: Selection strategy
            
        Returns:
            Dictionary mapping image counts to lists of selected images
        """
        if seed is not None:
            random.seed(seed)
            
        # Generate image counts for each set
        counts = self.generate_image_counts(num_sets, max_images, scaling)
        print(f"Creating {len(counts)} image sets with counts: {counts}")
        
        # Get the full list to select from
        all_images = self.image_list.copy()
        random.shuffle(all_images)
        
        # Create incremental sets
        sets = {}
        for count in counts:
            sets[count] = all_images[:count]
            
        return sets
        
    def export_image_sets(self, incremental_sets, pipeline_config_template, train_config_template):
        """
        Export the selected image sets and create configuration files for processing and training.
        
        Args:
            incremental_sets: Dictionary mapping image counts to lists of selected images
            pipeline_config_template: Path to pipeline configuration template
            train_config_template: Path to training configuration template
            
        Returns:
            Dictionary with paths to the exported dataset configurations
        """
        # Load template configurations
        with open(pipeline_config_template, 'r') as f:
            pipeline_config = yaml.safe_load(f)
            
        with open(train_config_template, 'r') as f:
            train_config = yaml.safe_load(f)
            
        # Create experiment directory
        experiment_dir = self.output_base_path / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directories for each set
        output_configs = {}
        for count, images in incremental_sets.items():
            # Create set directory
            set_dir = experiment_dir / f"set_{count}"
            set_dir.mkdir(exist_ok=True)
            
            # Create image subset directory
            images_dir = set_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            # Create labels directory
            labels_dir = set_dir / "labels"
            labels_dir.mkdir(exist_ok=True)
            
            # Copy selected images and their corresponding labels
            image_path = self.base_path / self.data_config['data'][0]
            label_path = Path(str(image_path).replace('/images', '/labels'))
            
            for img in images:
                # Copy image
                shutil.copy(image_path / img, images_dir / img)
                
                # Copy label (if exists)
                label_file = img.rsplit('.', 1)[0] + '.txt'
                if (label_path / label_file).exists():
                    shutil.copy(label_path / label_file, labels_dir / label_file)
                else:
                    print(f"Warning: Label not found for {img}")
                    
            # Create YAML configuration for this subset
            subset_data_yaml = set_dir / "cgras_data.yaml"
            subset_config = {
                'names': self.data_config['names'],
                'path': str(set_dir),
                'data': ['images'],
                'labels': ['labels']
            }
            
            with open(subset_data_yaml, 'w') as f:
                yaml.dump(subset_config, f)
                
            # Create pipeline config for this subset
            pipeline_config_path = set_dir / "pipeline_config.yaml"
            subset_pipeline_config = pipeline_config.copy()
            subset_pipeline_config['project_name'] = f"{self.experiment_name}_{count}"
            subset_pipeline_config['input_path'] = str(set_dir)
            subset_pipeline_config['output_base_path'] = str(experiment_dir / "outputs")
            
            with open(pipeline_config_path, 'w') as f:
                yaml.dump(subset_pipeline_config, f)
                
            # Create training config for this subset
            train_config_path = set_dir / "train_config.yaml"
            subset_train_config = train_config.copy()
            subset_train_config['name'] = f"train_{self.experiment_name}_{count}"
            # The yaml_path will be updated after pipeline processing
            processed_data_path = experiment_dir / "outputs" / f"{self.experiment_name}_{count}_filtered_split_tiled_balanced" / "cgras_data.yaml"
            subset_train_config['yaml_path'] = str(processed_data_path)
            
            with open(train_config_path, 'w') as f:
                yaml.dump(subset_train_config, f)
                
            # Store config paths for return
            output_configs[count] = {
                'data_yaml': subset_data_yaml,
                'pipeline_config': pipeline_config_path,
                'train_config': train_config_path,
                'processed_yaml_path': processed_data_path
            }
            
        return output_configs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Select image subsets for incremental training.")
    parser.add_argument('--data_yaml', required=True, help='Path to the cgras_data.yaml file')
    parser.add_argument('--output_path', required=True, help='Base path for outputs')
    parser.add_argument('--pipeline_config', required=True, help='Path to pipeline config template')
    parser.add_argument('--train_config', required=True, help='Path to training config template')
    parser.add_argument('--experiment_name', default='CoralScaling', help='Name of the experiment')
    parser.add_argument('--num_sets', type=int, default=10, help='Number of different sized sets to create')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to use')
    parser.add_argument('--scaling', choices=['log', 'linear'], default='log', help='Scaling of image counts')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--strategy', choices=['random', 'stratified_week', 'stratified_tile'], 
                        default='random', help='Selection strategy')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize selector
    selector = ImageSelector(args.data_yaml, args.output_path, args.experiment_name)
    
    # Create incremental sets
    incremental_sets = selector.create_incremental_sets(
        num_sets=args.num_sets,
        max_images=args.max_images,
        scaling=args.scaling,
        seed=args.seed,
        strategy=args.strategy
    )
    
    # Export sets and create configs
    output_configs = selector.export_image_sets(
        incremental_sets,
        args.pipeline_config,
        args.train_config
    )
    
    print(f"Created {len(incremental_sets)} image sets for experiment '{args.experiment_name}'")
    for count, config in output_configs.items():
        print(f"Set with {count} images: {config['data_yaml']}")