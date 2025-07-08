#! /usr/bin/env python3

import os
import re
import yaml
import pulp
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt


class DatasetSplitter:
    """
    A class to split YOLO datasets into train, validation, and test sets.
    Specifically designed for CGRAS datasets with specific file naming structure.
    """
    
    def __init__(self, yaml_path, output_path=None):
        """
        Initialize the DatasetSplitter with paths.
        
        Args:
            yaml_path (str): Path to the YOLO data YAML file
            output_path (str, optional): Path where split data should be saved
        """
        self.yaml_path = Path(yaml_path)
        self.output_path = Path(output_path) if output_path else self.yaml_path.parent / "split_data"
        self.yaml_data = None
        self.image_paths = []
        self.label_paths = []
        self.file_info = []  # Stores parsed file information
        self.max_workers = min(32, os.cpu_count() + 4)
        self.new_yaml_path = None
        
        # Regex pattern for CGRAS file naming convention
        # CGRAS_<Species>_<Room>_<date>_w<week_number>_T<tile_number>_<index_number>.jpg
        self.cgras_pattern = re.compile(
            r'CGRAS_([^_]+)_([^_]+)_(\d{8})_w(\d+)_T(\d{2})_(\d{2})\.([^.]+)$'
        )
        # ASPA pattern: <number>-<field1>-<field2>-<field3>-<field4>-<date>-<time>.jpg
        self.aspa_pattern = re.compile(
            r'(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d{6})-(\d{4})\.([^.]+)$'
        )
        
        # Load YAML data
        self._load_yaml()
        
    def _load_yaml(self):
        """Load the YAML configuration file and find all image paths"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found at {self.yaml_path}")
            
        with open(self.yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            
        self.base_dir = self.yaml_path.parent
        self.image_paths = []
        self.label_paths = []
        
        # Extract data paths from YAML
        data_paths = []
        if 'data' in self.yaml_data and isinstance(self.yaml_data['data'], list):
            data_paths = self.yaml_data['data']
        elif 'data' in self.yaml_data and isinstance(self.yaml_data['data'], dict):
            for dataset_name, dataset_info in self.yaml_data['data'].items():
                if 'images' in dataset_info:
                    data_paths.append(dataset_info['images'])
        else:
            raise ValueError("Unsupported YAML format: 'data' field not found or in unexpected format")
        
        print(f"Found {len(data_paths)} dataset paths in the YAML file")
        
        # Find all image files in the data paths
        for data_path in data_paths:
            images_dir = self.base_dir / data_path
            if not images_dir.exists():
                print(f"Warning: Directory not found: {images_dir}")
                continue
                
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                found_images = list(images_dir.glob(f"**/*{ext}"))
                self.image_paths.extend(found_images)
                
                # Find corresponding label files
                for img_path in found_images:
                    label_path = self._find_label_path(img_path)
                    if label_path.exists():
                        self.label_paths.append(label_path)
                    else:
                        print(f"Warning: No label file found for {img_path}")
        
        print(f"Found {len(self.image_paths)} images and {len(self.label_paths)} label files")
    
    def _find_label_path(self, image_path):
        """
        For a given image path, determine the corresponding label path.
        
        Args:
            image_path (Path): Path to an image
            
        Returns:
            Path: Path to the corresponding label file
        """
        # Convert image path to label path
        parent_dir = image_path.parent
        
        if 'images' in str(parent_dir):
            label_dir = str(parent_dir).replace('images', 'labels')
            label_path = Path(label_dir) / f"{image_path.stem}.txt"
            return label_path
            
        # If not found, try one level up
        if 'images' in str(parent_dir.parent):
            label_dir = str(parent_dir.parent).replace('images', 'labels')
            label_path = Path(label_dir) / f"{image_path.stem}.txt"
            return label_path
            
        # Default to same directory but with .txt extension
        return image_path.with_suffix('.txt')
    
    def parse_file_info(self):
        """
        Parse all image filenames to extract metadata according to the CGRAS or ASPA naming convention.
        
        Returns:
            pandas.DataFrame: DataFrame containing parsed file information
        """
        file_info = []
        malformed_files = []
        
        print("Parsing file information...")
        for img_path in tqdm(self.image_paths, unit="files"):
            filename = img_path.name
            
            # Try CGRAS pattern first
            cgras_match = self.cgras_pattern.search(filename)
            aspa_match = self.aspa_pattern.search(filename)
            
            if cgras_match:
                species, room, date, week, tile, index, ext = cgras_match.groups()
                
                # Convert to appropriate types
                week = int(week)
                tile = int(tile)
                index = int(index)
                
                # Find label path
                label_path = self._find_label_path(img_path)
                has_label = label_path.exists()
                
                file_info.append({
                    'filename': filename,
                    'species': species,
                    'room': room,
                    'date': date,
                    'week': week,
                    'tile': tile,
                    'index': index,
                    'extension': ext,
                    'image_path': img_path,
                    'label_path': label_path if has_label else None,
                    'has_label': has_label,
                    'original_dataset': self._get_dataset_name(img_path)
                })
            elif aspa_match:
                id_num, field1, field2, field3, field4, date, time, ext = aspa_match.groups()
                
                # Convert to appropriate types
                id_num = int(id_num)
                field1 = int(field1)
                field2 = int(field2) 
                field3 = int(field3)
                field4 = int(field4)
                
                # Find label path
                label_path = self._find_label_path(img_path)
                has_label = label_path.exists()
                
                # For ASPA files, we'll use field1 as "room" and field3 as "tile" to maintain compatibility
                file_info.append({
                    'filename': filename,
                    'species': 'aspa',  # Default species for ASPA files
                    'room': f"room_{field1}",  # Use field1 as room identifier
                    'date': date,
                    'week': 1,  # Default week for ASPA files
                    'tile': field3,  # Use field3 as tile
                    'index': id_num,  # Use the ID number as index
                    'extension': ext,
                    'aspa_field1': field1,  # Store original ASPA fields
                    'aspa_field2': field2,
                    'aspa_field3': field3,
                    'aspa_field4': field4,
                    'aspa_time': time,
                    'image_path': img_path,
                    'label_path': label_path if has_label else None,
                    'has_label': has_label,
                    'original_dataset': self._get_dataset_name(img_path)
                })
            else:
                malformed_files.append(filename)
        
        if malformed_files:
            print(f"Warning: {len(malformed_files)} files don't match the expected naming pattern.")
            if len(malformed_files) <= 10:
                print("Malformed files:")
                for f in malformed_files:
                    print(f"  - {f}")
            else:
                print("First 10 malformed files:")
                for f in malformed_files[:10]:
                    print(f"  - {f}")
                print(f"  ... and {len(malformed_files) - 10} more")
        
        self.file_info = pd.DataFrame(file_info)
        print(f"Successfully parsed {len(self.file_info)} files")
        return self.file_info
    
    def _get_dataset_name(self, file_path):
        """Extract the dataset name from a file path based on YAML paths"""
        for data_path in self.yaml_data.get('data', []):
            data_dir = self.base_dir / data_path
            if data_dir in file_path.parents:
                # Extract dataset name from data_path
                if '/' in data_path:
                    parts = data_path.split('/')
                    return parts[1] if len(parts) > 1 else parts[0]  # datasets/dataset1/data/images -> dataset1
                return data_path
        return "unknown"
    
    def count_by_field(self, field):
        """
        Count images by a specific field.
        
        Args:
            field (str): Field to count by ('species', 'room', 'week', 'tile', etc.)
            
        Returns:
            dict: Counts for each unique value in the field
        """
        if self.file_info is None or len(self.file_info) == 0:
            print("No file information available. Running parse_file_info first.")
            self.parse_file_info()
        
        if field not in self.file_info.columns:
            valid_fields = [col for col in self.file_info.columns if col not in 
                           ['filename', 'image_path', 'label_path', 'has_label']]
            print(f"Invalid field '{field}'. Valid fields are: {', '.join(valid_fields)}")
            return None
        
        counts = self.file_info[field].value_counts().to_dict()
        
        # Print the counts
        print(f"\nCounts by {field}:")
        for value, count in sorted(counts.items()):
            print(f"  - {field}={value}: {count} images")
        
        # Create a bar plot of the counts
        plt.figure(figsize=(10, 6))
        plt.bar(counts.keys(), counts.values())
        plt.title(f'Image Count by {field}')
        plt.xlabel(field)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return counts
    
    def visualize_distribution(self, fields=None):
        """
        Visualize the distribution of images across different fields.
        
        Args:
            fields (list, optional): List of fields to visualize. If None, uses common fields.
            
        Returns:
            None
        """
        if self.file_info is None or len(self.file_info) == 0:
            print("No file information available. Running parse_file_info first.")
            self.parse_file_info()
            
        if fields is None:
            fields = ['species', 'room', 'week', 'tile', 'original_dataset']
            
        for field in fields:
            if field not in self.file_info.columns:
                print(f"Field '{field}' not found. Skipping.")
                continue
                
            self.count_by_field(field)
    
    def preview_split(self, split_field="tile", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                      stratify_by=None, random_seed=42):
        """
        Preview split distribution using ILP optimization, without modifying any files or assignments.

        Args:
            split_field (str): Field to split by
            train_ratio (float): Desired training split
            val_ratio (float): Desired validation split
            test_ratio (float): Desired test split
            stratify_by (str, optional): Field to stratify by
            random_seed (int): Random seed for reproducibility

        Returns:
            dict: Proposed split assignments
        """
        if self.file_info is None or len(self.file_info) == 0:
            self.parse_file_info()

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        if split_field is None:
            print("No split_field provided. Previewing random image-level split.")

            shuffled = self.file_info.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            total = len(shuffled)
            train_end = int(train_ratio * total)
            val_end = train_end + int(val_ratio * total)

            split_assignment = pd.Series(index=shuffled.index, dtype=object)
            split_assignment.iloc[:train_end] = 'train'
            split_assignment.iloc[train_end:val_end] = 'val'
            split_assignment.iloc[val_end:] = 'test'

            split_counts = split_assignment.value_counts().to_dict()

            # Plot 1: Actual split counts
            plt.figure(figsize=(8, 4))
            plt.bar(split_counts.keys(), split_counts.values())
            plt.title("Proposed Random Split Counts")
            plt.ylabel("Number of Items")
            plt.tight_layout()
            plt.show()

            # Plot 2: Difference from target
            actual_ratios = {k: v / total for k, v in split_counts.items()}
            target_ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
            diff = {k: abs(actual_ratios.get(k, 0) - target_ratios[k]) for k in target_ratios}

            plt.figure(figsize=(8, 4))
            plt.bar(diff.keys(), diff.values())
            plt.title("Absolute Deviation from Target Ratios")
            plt.ylabel("Deviation")
            plt.tight_layout()
            plt.show()

            return split_assignment.to_dict()

        random.seed(random_seed)
        np.random.seed(random_seed)

        unique_values = self.file_info[split_field].unique()
        global_value_counts = {}

        if stratify_by and stratify_by != split_field:
            assignments = {}
            groups = self.file_info.groupby(stratify_by)
            for _, group_df in groups:
                group_values = group_df[split_field].unique()
                value_counts = {val: len(group_df[group_df[split_field] == val]) for val in group_values}
                global_value_counts.update(value_counts)
                total = sum(value_counts.values())
                target_train = total * train_ratio
                target_val = total * val_ratio
                target_test = total * test_ratio
                partial = self._optimize_split(value_counts, target_train, target_val, target_test)
                assignments.update(partial)
        else:
            value_counts = {val: len(self.file_info[self.file_info[split_field] == val]) for val in unique_values}
            global_value_counts = value_counts
            total = sum(value_counts.values())
            target_train = total * train_ratio
            target_val = total * val_ratio
            target_test = total * test_ratio
            assignments = self._optimize_split(value_counts, target_train, target_val, target_test)

        # Count totals for each split
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        for val, split in assignments.items():
            split_counts[split] += global_value_counts[val]

        # Plot 1: Actual split counts
        plt.figure(figsize=(8, 4))
        plt.bar(split_counts.keys(), split_counts.values())
        plt.title("Proposed Split Counts")
        plt.ylabel("Number of Items")
        plt.tight_layout()
        plt.show()

        # Plot 2: Difference from target
        total = sum(split_counts.values())
        actual_ratios = {k: v / total for k, v in split_counts.items()}
        target_ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
        diff = {k: abs(actual_ratios[k] - target_ratios[k]) for k in split_counts}

        plt.figure(figsize=(8, 4))
        plt.bar(diff.keys(), diff.values())
        plt.title("Absolute Deviation from Target Ratios")
        plt.ylabel("Deviation")
        plt.tight_layout()
        plt.show()

        return assignments

    def create_splits(self, split_field=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                        stratify_by=None, random_seed=42):
        """
        Create train/validation/test splits based on a specific field.
        
        Args:
            split_field (str): Field to use for splitting ('tile', 'week', etc.)
            train_ratio (float): Proportion for training set
            val_ratio (float): Proportion for validation set
            test_ratio (float): Proportion for test set
            stratify_by (str, optional): Field to stratify by (e.g., 'species')
            random_seed (int): Random seed for reproducibility
            
        Returns:
            dict: Split assignments for each unique value in split_field
        """
        if self.file_info is None or len(self.file_info) == 0:
            print("No file information available. Running parse_file_info first.")
            self.parse_file_info()
            
        # Check that ratios sum to 1.0
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
            
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        total = len(self.file_info)
        
        # Initialize the split column with empty values
        self.file_info['split'] = np.nan
        
        # Special handling for very small datasets (less than 3 images)
        if total < 3:
            print(f"Warning: Very small dataset with only {total} images.")
            
            if total == 1:
                # With just 1 image, put it in training
                self.file_info['split'] = 'train'
                print("Only one image available - assigning to training set.")
            elif total == 2:
                # With 2 images, put one in training and one in validation
                self.file_info.iloc[0, self.file_info.columns.get_loc('split')] = 'train'
                self.file_info.iloc[1, self.file_info.columns.get_loc('split')] = 'val'
                print("Only two images available - assigning one to training and one to validation.")
        elif total == 3:
            # With exactly 3 images, put one in each split
            self.file_info.iloc[0, self.file_info.columns.get_loc('split')] = 'train'
            self.file_info.iloc[1, self.file_info.columns.get_loc('split')] = 'val'
            self.file_info.iloc[2, self.file_info.columns.get_loc('split')] = 'test'
            print("Exactly three images available - assigning one to each split.")
        else:
            # For larger datasets, use either field-based splitting or random splitting
            if split_field is None:
                print("No split_field provided. Performing random image-level split with guaranteed representation.")
                
                # Shuffle file_info
                self.file_info = self.file_info.sample(frac=1, random_state=random_seed).reset_index(drop=True)
                
                # Always ensure at least one image in each split
                self.file_info.iloc[0, self.file_info.columns.get_loc('split')] = 'train'
                self.file_info.iloc[1, self.file_info.columns.get_loc('split')] = 'val'
                self.file_info.iloc[2, self.file_info.columns.get_loc('split')] = 'test'
                
                # For remaining images, prioritize filling train, then val, then test
                remaining = total - 3
                
                # Calculate remaining counts based on the target ratios
                # Normalize the ratios to sum to 1
                ratio_sum = train_ratio + val_ratio + test_ratio
                train_norm = train_ratio / ratio_sum
                val_norm = val_ratio / ratio_sum
                test_norm = test_ratio / ratio_sum
                
                # Calculate how many images should go in each split from the remaining
                # Prioritize train > val > test when dealing with rounding
                train_remaining = int(remaining * train_norm)
                val_remaining = int(remaining * val_norm)
                test_remaining = remaining - train_remaining - val_remaining
                
                # Assign remaining images accordingly
                if train_remaining > 0:
                    self.file_info.iloc[3:3+train_remaining, self.file_info.columns.get_loc('split')] = 'train'
                
                if val_remaining > 0:
                    self.file_info.iloc[3+train_remaining:3+train_remaining+val_remaining, self.file_info.columns.get_loc('split')] = 'val'
                
                if test_remaining > 0:
                    self.file_info.iloc[3+train_remaining+val_remaining:, self.file_info.columns.get_loc('split')] = 'test'
            else:
                # Regular field-based splitting
                # Get all unique values for the split field
                unique_values = self.file_info[split_field].unique()
                
                # Initialize split assignments
                split_assignments = {}
                
                if stratify_by is not None and stratify_by != split_field:
                    # Create stratified splits
                    print(f"Creating stratified splits using {split_field}, stratified by {stratify_by}")
                    
                    # Group by stratify field
                    groups = self.file_info.groupby(stratify_by)
                    
                    for group_name, group_df in groups:
                        # Get unique split values for this group
                        group_values = group_df[split_field].unique()
                        
                        # Calculate counts for each split value
                        value_counts = {}
                        for val in group_values:
                            value_counts[val] = len(group_df[group_df[split_field] == val])
                        
                        # Calculate total items in this group
                        total_count = sum(value_counts.values())
                        
                        # Calculate target counts for each split within this group
                        target_train = total_count * train_ratio
                        target_val = total_count * val_ratio
                        target_test = total_count * test_ratio
                        
                        # Optimize the allocation to hit these targets
                        split_result = self._optimize_split(
                            value_counts, target_train, target_val, target_test
                        )
                        
                        # Update the overall split assignments
                        for val, split in split_result.items():
                            split_assignments[val] = split
                            
                        # Print the result for this group
                        train_actual = sum(value_counts[v] for v, s in split_result.items() if s == 'train')
                        val_actual = sum(value_counts[v] for v, s in split_result.items() if s == 'val')
                        test_actual = sum(value_counts[v] for v, s in split_result.items() if s == 'test')
                        
                        print(f"Group {stratify_by}={group_name}:")
                        print(f"  - Target:  train={target_train:.1f}, val={target_val:.1f}, test={target_test:.1f}")
                        print(f"  - Actual:  train={train_actual} ({train_actual/total_count:.1%}), "
                            f"val={val_actual} ({val_actual/total_count:.1%}), "
                            f"test={test_actual} ({test_actual/total_count:.1%})")
                else:
                    # Create non-stratified splits
                    print(f"Creating non-stratified splits using {split_field}")
                    
                    # Calculate counts for each split value
                    value_counts = {}
                    for val in unique_values:
                        value_counts[val] = len(self.file_info[self.file_info[split_field] == val])
                        
                    # Calculate total items
                    total_count = sum(value_counts.values())
                    
                    # Calculate target counts
                    target_train = total_count * train_ratio
                    target_val = total_count * val_ratio
                    target_test = total_count * test_ratio
                    
                    # Optimize the allocation to hit these targets
                    split_assignments = self._optimize_split(
                        value_counts, target_train, target_val, target_test
                    )
                    
                    # Print the results
                    train_actual = sum(value_counts[v] for v, s in split_assignments.items() if s == 'train')
                    val_actual = sum(value_counts[v] for v, s in split_assignments.items() if s == 'val')
                    test_actual = sum(value_counts[v] for v, s in split_assignments.items() if s == 'test')
                    
                    print("Overall split:")
                    print(f"  - Target:  train={target_train:.1f} ({train_ratio:.1%}), "
                        f"val={target_val:.1f} ({val_ratio:.1%}), "
                        f"test={target_test:.1f} ({test_ratio:.1%})")
                    print(f"  - Actual:  train={train_actual} ({train_actual/total_count:.1%}), "
                        f"val={val_actual} ({val_actual/total_count:.1%}), "
                        f"test={test_actual} ({test_actual/total_count:.1%})")
                
                # Assign a split to each file based on split_field
                self.file_info['split'] = self.file_info[split_field].map(split_assignments)
        
            # Print counts per split
            train_count = (self.file_info['split'] == 'train').sum()
            val_count = (self.file_info['split'] == 'val').sum()
            test_count = (self.file_info['split'] == 'test').sum()
            total = len(self.file_info)
            
            print(f"\nFinal Split Counts:")
            print(f"  - Train: {train_count} ({train_count/total:.1%})")
            print(f"  - Validation: {val_count} ({val_count/total:.1%})")
            print(f"  - Test: {test_count} ({test_count/total:.1%})")
            
            return None if split_field is None else split_assignments


    def _optimize_split(self, value_counts, target_train, target_val, target_test):
        """
        Optimize the allocation of values to train/val/test using Integer Linear Programming (ILP).
        For small datasets, ensures each split has at least one sample with priority to train > val > test.

        Args:
            value_counts (dict): Counts for each value
            target_train (float): Target count for training set
            target_val (float): Target count for validation set
            target_test (float): Target count for test set

        Returns:
            dict: Mapping of values to 'train', 'val', or 'test'
        """
        values = list(value_counts.keys())
        counts = value_counts
        total_samples = sum(counts.values())
        
        print(f"Total samples: {total_samples}, Unique values: {len(values)}")
        
        # Special handling for very small datasets
        if len(values) < 3:
            allocation = {}
            # Sort by sample count in descending order
            sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            
            # If only 1 or 2 values, prioritize train > val > test
            if len(values) == 1:
                allocation[values[0]] = 'train'
                print(f"Warning: Only one value ({values[0]}) available. Assigning to training set only.")
                return allocation
            elif len(values) == 2:
                # Assign the one with more samples to train, the other to val
                allocation[sorted_values[0][0]] = 'train'
                allocation[sorted_values[1][0]] = 'val'
                print(f"Warning: Only two values available. No test set will be created.")
                return allocation
        
        # Handle case where we have exactly 3 values
        if len(values) == 3:
            allocation = {}
            # Simply assign one to each split, prioritizing by sample count
            sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            allocation[sorted_values[0][0]] = 'train'
            allocation[sorted_values[1][0]] = 'val'
            allocation[sorted_values[2][0]] = 'test'
            print("Exactly 3 values available. Assigned one to each split, prioritizing by sample count.")
            return allocation

        # Standard ILP optimization for larger datasets
        prob = pulp.LpProblem("DatasetSplitOptimization", pulp.LpMinimize)

        # Decision variables: assignment of each value to one split
        x_train = {val: pulp.LpVariable(f"x_train_{val}", cat="Binary") for val in values}
        x_val = {val: pulp.LpVariable(f"x_val_{val}", cat="Binary") for val in values}
        x_test = {val: pulp.LpVariable(f"x_test_{val}", cat="Binary") for val in values}

        # Ensure each value is assigned to exactly one split
        for val in values:
            prob += x_train[val] + x_val[val] + x_test[val] == 1

        # Total counts in each split
        total_train = pulp.lpSum([x_train[val] * counts[val] for val in values])
        total_val = pulp.lpSum([x_val[val] * counts[val] for val in values])
        total_test = pulp.lpSum([x_test[val] * counts[val] for val in values])

        # Add absolute deviation variables
        dev_train = pulp.LpVariable("dev_train", lowBound=0)
        dev_val = pulp.LpVariable("dev_val", lowBound=0)
        dev_test = pulp.LpVariable("dev_test", lowBound=0)

        # Deviation constraints (absolute value modeled with two inequalities)
        prob += total_train - target_train <= dev_train
        prob += target_train - total_train <= dev_train

        prob += total_val - target_val <= dev_val
        prob += target_val - total_val <= dev_val

        prob += total_test - target_test <= dev_test
        prob += target_test - total_test <= dev_test
        
        # Ensure at least one sample in each split if we have enough values
        if len(values) >= 3:
            prob += total_train >= 1
            prob += total_val >= 1
            prob += total_test >= 1

        # Objective: minimize sum of deviations with weighted priorities (train > val > test)
        # We use weights to prioritize filling train first, then val, then test
        # Note that since we're minimizing deviations, a larger weight means higher priority
        # to minimize that deviation (i.e., to get closer to the target)
        train_weight = 3  # Highest priority
        val_weight = 2    # Medium priority
        test_weight = 1   # Lowest priority
        
        prob += train_weight * dev_train + val_weight * dev_val + test_weight * dev_test

        # Solve
        status = prob.solve()
        if pulp.LpStatus[status] != "Optimal":
            print("Warning: ILP did not find an optimal solution. Falling back to greedy.")
            return self._optimize_split_greedy(value_counts, target_train, target_val, target_test)

        # Create allocation
        allocation = {}
        for val in values:
            if pulp.value(x_train[val]) == 1:
                allocation[val] = 'train'
            elif pulp.value(x_val[val]) == 1:
                allocation[val] = 'val'
            elif pulp.value(x_test[val]) == 1:
                allocation[val] = 'test'

        # Verify that each split has at least one sample if we have enough values
        if len(values) >= 3:
            # Get counts for each split
            train_actual = sum(counts[v] for v, s in allocation.items() if s == 'train')
            val_actual = sum(counts[v] for v, s in allocation.items() if s == 'val')
            test_actual = sum(counts[v] for v, s in allocation.items() if s == 'test')
            
            # Check if any split is empty
            if train_actual == 0 or val_actual == 0 or test_actual == 0:
                print("Warning: ILP solution resulted in an empty split. Falling back to greedy with strict constraints.")
                return self._optimize_split_greedy_with_constraints(value_counts, target_train, target_val, target_test)

        return allocation

    def _optimize_split_greedy(self, value_counts, target_train, target_val, target_test):
        """
        Greedy allocation with priority to fill train, then val, then test.
        
        Args:
            value_counts (dict): Counts for each value
            target_train (float): Target count for training set
            target_val (float): Target count for validation set
            target_test (float): Target count for test set
            
        Returns:
            dict: Mapping of values to 'train', 'val', or 'test'
        """
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        allocation = {}
        current_train = 0
        current_val = 0
        current_test = 0
        
        # Special handling for very small datasets
        if len(sorted_values) < 3:
            if len(sorted_values) == 1:
                allocation[sorted_values[0][0]] = 'train'
                print(f"Warning: Only one value available. Assigning to training set only.")
                return allocation
            elif len(sorted_values) == 2:
                # Assign the one with more samples to train, the other to val
                allocation[sorted_values[0][0]] = 'train'
                allocation[sorted_values[1][0]] = 'val'
                print(f"Warning: Only two values available. No test set will be created.")
                return allocation
        
        # Handle case where we have exactly 3 values
        if len(sorted_values) == 3:
            allocation[sorted_values[0][0]] = 'train'
            allocation[sorted_values[1][0]] = 'val'
            allocation[sorted_values[2][0]] = 'test'
            print("Exactly 3 values available. Assigned one to each split, prioritizing by sample count.")
            return allocation
        
        # Make sure we have at least one sample in each split
        if len(sorted_values) >= 3:
            # First, allocate at least one value to each split, prioritizing by count
            allocation[sorted_values[0][0]] = 'train'
            current_train += sorted_values[0][1]
            
            allocation[sorted_values[1][0]] = 'val'
            current_val += sorted_values[1][1]
            
            allocation[sorted_values[2][0]] = 'test'
            current_test += sorted_values[2][1]
            
            # Skip the first 3 in the loop below
            sorted_values = sorted_values[3:]

        # For remaining values, use the standard greedy approach with priority weighting
        for val, count in sorted_values:
            # Apply weights to prioritize train > val > test
            # Lower weighted difference means higher priority
            train_weight = 0.8  # Priority to fill train first
            val_weight = 0.9    # Second priority
            test_weight = 1.0   # Lowest priority
            
            # Calculate weighted differences from target
            train_diff = train_weight * abs((current_train + count) / max(1, target_train) - 1)
            val_diff = val_weight * abs((current_val + count) / max(1, target_val) - 1)
            test_diff = test_weight * abs((current_test + count) / max(1, target_test) - 1)

            min_diff = min(train_diff, val_diff, test_diff)

            if min_diff == train_diff:
                allocation[val] = 'train'
                current_train += count
            elif min_diff == val_diff:
                allocation[val] = 'val'
                current_val += count
            else:
                allocation[val] = 'test'
                current_test += count

        return allocation

    def _optimize_split_greedy_with_constraints(self, value_counts, target_train, target_val, target_test):
        """
        Enhanced greedy algorithm that guarantees at least one sample in each split.
        
        Args:
            value_counts (dict): Counts for each value
            target_train (float): Target count for training set
            target_val (float): Target count for validation set
            target_test (float): Target count for test set
            
        Returns:
            dict: Mapping of values to 'train', 'val', or 'test'
        """
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        allocation = {}
        
        # First, ensure each split has at least one value
        if len(sorted_values) >= 3:
            allocation[sorted_values[0][0]] = 'train'
            allocation[sorted_values[1][0]] = 'val'
            allocation[sorted_values[2][0]] = 'test'
            
            current_train = sorted_values[0][1]
            current_val = sorted_values[1][1]
            current_test = sorted_values[2][1]
            
            # Process the remaining values
            for i in range(3, len(sorted_values)):
                val, count = sorted_values[i]
                
                # Apply weights to prioritize train > val > test
                train_weight = 0.8  # Priority to fill train first
                val_weight = 0.9    # Second priority
                test_weight = 1.0   # Lowest priority
                
                train_diff = train_weight * abs((current_train + count) / max(1, target_train) - 1)
                val_diff = val_weight * abs((current_val + count) / max(1, target_val) - 1)
                test_diff = test_weight * abs((current_test + count) / max(1, target_test) - 1)
                
                min_diff = min(train_diff, val_diff, test_diff)
                
                if min_diff == train_diff:
                    allocation[val] = 'train'
                    current_train += count
                elif min_diff == val_diff:
                    allocation[val] = 'val'
                    current_val += count
                else:
                    allocation[val] = 'test'
                    current_test += count
        else:
            # For very small datasets, use the standard method
            return self._optimize_split_greedy(value_counts, target_train, target_val, target_test)
        
        return allocation
    
    def export_splits(self):
        """
        Export the dataset splits to the output directory with desired structure.
        
        Returns:
            bool: True if successful
        """
        if 'split' not in self.file_info.columns:
            print("No split information available. Run create_splits first.")
            return False
            
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create split directories
        split_dirs = {
            'train': self.output_path / 'train',
            'val': self.output_path / 'valid',
            'test': self.output_path / 'test'
        }
        
        for split_dir in split_dirs.values():
            split_dir.mkdir(exist_ok=True)
        
        # Group files by original dataset
        datasets = self.file_info['original_dataset'].unique()
        
        # Copy files to their respective split directories
        with tqdm(total=len(self.file_info), desc="Copying files", unit="files") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for _, row in self.file_info.iterrows():
                    split = row['split']
                    dataset_name = row['original_dataset']
                    
                    # Create dataset & images subdirectory in split
                    (split_dirs[split] / 'images').mkdir(parents=True, exist_ok=True)
                    (split_dirs[split] / 'labels').mkdir(parents=True, exist_ok=True)

                    
                    # Source paths
                    src_img_path = row['image_path']
                    src_label_path = row['label_path']
                    
                    # Destination paths - update these paths:
                    dst_img_path = split_dirs[split] / 'images' / src_img_path.name
                    dst_label_path = split_dirs[split] / 'labels' / (src_img_path.stem + '.txt')

                    
                    # Copy image
                    future1 = executor.submit(shutil.copy2, src_img_path, dst_img_path)
                    futures.append(future1)
                    
                    # Copy label if it exists
                    if src_label_path and Path(src_label_path).exists():
                        future2 = executor.submit(shutil.copy2, src_label_path, dst_label_path)
                        futures.append(future2)
                    
                    pbar.update(1)
                
                # Wait for all copy operations to complete
                concurrent.futures.wait(futures)
        
        # Create YAML file with the new split structure
        yaml_data = {
            'path': str(self.output_path.absolute()),
            'train': ['train/images'],
            'val': ['valid/images'],
            'test': ['test/images'],
            'names': self.yaml_data['names']
        }
        
        # Write the YAML file
        yaml_path = self.output_path / 'cgras_data.yaml'
        self.new_yaml_path = yaml_path
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            
        print(f"Exported splits to {self.output_path}")
        print(f"Created YAML file at {yaml_path}")
        
        # Print summary statistics
        train_count = (self.file_info['split'] == 'train').sum()
        val_count = (self.file_info['split'] == 'val').sum()
        test_count = (self.file_info['split'] == 'test').sum()
        total = len(self.file_info)
        
        print("\nExport Summary:")
        print(f"  - Train: {train_count} images ({train_count/total:.1%})")
        print(f"  - Validation: {val_count} images ({val_count/total:.1%})")
        print(f"  - Test: {test_count} images ({test_count/total:.1%})")
        
        return True
    
    def _get_split_paths(self, split, datasets):
        """
        Generate the list of dataset paths for a specific split with the desired structure.
        
        Args:
            split (str): Split name ('train', 'val', or 'test')
            datasets (list): List of dataset names
            
        Returns:
            list: List of dataset paths for the split
        """
        split_dir = 'valid' if split == 'val' else split
        # Change the path format here:
        return [f"{split_dir}/{dataset}/images" for dataset in datasets]


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split YOLO datasets for training, validation, and testing")
    parser.add_argument("yaml_path", help="Path to the YOLO data YAML file")
    parser.add_argument("--output", help="Path where split data should be saved")
    parser.add_argument("--split-by", default="tile", help="Field to split by (default: tile)")
    parser.add_argument("--stratify-by", help="Field to stratify by (e.g., species)")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio (default: 0.7)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--analyze", action="store_true", help="Only analyze distribution without splitting")
    parser.add_argument("--visualize-field", help="Field to visualize distribution (e.g., species, tile, week)")
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(args.yaml_path, args.output)
    splitter.parse_file_info()
    
    if args.analyze:
        splitter.visualize_distribution()
    elif args.visualize_field:
        splitter.count_by_field(args.visualize_field)
    else:
        splitter.create_splits(
            split_field=args.split_by,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            stratify_by=args.stratify_by,
            random_seed=args.seed
        )
        splitter.export_splits()