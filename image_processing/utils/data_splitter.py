import os
import re
import yaml
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
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
        
        # Regex pattern for CGRAS file naming convention
        # CGRAS_<Species>_<Room>_<date>_w<week_number>_T<tile_number>_<index_number>.jpg
        self.file_pattern = re.compile(
            r'CGRAS_([^_]+)_([^_]+)_(\d{8})_w(\d+)_T(\d{2})_(\d{2})\.([^.]+)$'
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
        Parse all image filenames to extract metadata according to the CGRAS naming convention.
        
        Returns:
            pandas.DataFrame: DataFrame containing parsed file information
        """
        file_info = []
        malformed_files = []
        
        print("Parsing file information...")
        for img_path in tqdm(self.image_paths, unit="files"):
            filename = img_path.name
            match = self.file_pattern.search(filename)
            
            if match:
                species, room, date, week, tile, index, ext = match.groups()
                
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
    
    def create_splits(self, split_field='tile', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
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
            
        if split_field not in self.file_info.columns:
            valid_fields = [col for col in self.file_info.columns if col not in 
                           ['filename', 'image_path', 'label_path', 'has_label']]
            print(f"Invalid split_field '{split_field}'. Valid fields are: {', '.join(valid_fields)}")
            return None
            
        # Get all unique values for the split field
        unique_values = self.file_info[split_field].unique()
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
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
        
        # Assign a split to each file
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
        
        return split_assignments
    
    def _optimize_split(self, value_counts, target_train, target_val, target_test):
        """
        Optimize the allocation of values to train/val/test to match target proportions.
        
        Args:
            value_counts (dict): Counts for each value
            target_train (float): Target count for training set
            target_val (float): Target count for validation set
            target_test (float): Target count for test set
            
        Returns:
            dict: Mapping of values to 'train', 'val', or 'test'
        """
        # Sort values by count (descending)
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Initialize allocation
        allocation = {}
        current_train = 0
        current_val = 0
        current_test = 0
        
        # First pass: allocate largest groups optimally
        for val, count in sorted_values:
            # Calculate which split would bring us closest to the target
            train_diff = abs((current_train + count) / target_train - 1)
            val_diff = abs((current_val + count) / target_val - 1)
            test_diff = abs((current_test + count) / target_test - 1)
            
            # Assign to the split that minimizes the difference
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
        
        # Optional: Second pass to further optimize if needed
        # This could shuffle some allocations to get even closer to the targets
        
        return allocation
    
    def export_splits(self):
        """
        Export the dataset splits to the output directory.
        
        Returns:
            bool: True if successful
        """
        if 'split' not in self.file_info.columns:
            print("No split information available. Run create_splits first.")
            return False
            
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create subdirectories for each split
        split_dirs = {
            'train': self.output_path / 'train',
            'val': self.output_path / 'valid',
            'test': self.output_path / 'test'
        }
        
        for split_name, split_dir in split_dirs.items():
            # Create images and labels directories
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Group files by original dataset
        datasets = self.file_info['original_dataset'].unique()
        
        # Copy files to their respective split directories
        with tqdm(total=len(self.file_info), desc="Copying files", unit="files") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for _, row in self.file_info.iterrows():
                    split = row['split']
                    dataset_name = row['original_dataset']
                    
                    # Create dataset subdirectory in split
                    (split_dirs[split] / 'images' / dataset_name).mkdir(exist_ok=True)
                    (split_dirs[split] / 'labels' / dataset_name).mkdir(exist_ok=True)
                    
                    # Source paths
                    src_img_path = row['image_path']
                    src_label_path = row['label_path']
                    
                    # Destination paths
                    dst_img_path = split_dirs[split] / 'images' / dataset_name / src_img_path.name
                    dst_label_path = split_dirs[split] / 'labels' / dataset_name / (src_img_path.stem + '.txt')
                    
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
            'train': self._get_split_paths('train', datasets),
            'val': self._get_split_paths('val', datasets),
            'test': self._get_split_paths('test', datasets),
            'names': self.yaml_data['names']  # Copy class names from original YAML
        }
        
        # Write the YAML file
        yaml_path = self.output_path / 'data.yaml'
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
        Generate the list of dataset paths for a specific split.
        
        Args:
            split (str): Split name ('train', 'val', or 'test')
            datasets (list): List of dataset names
            
        Returns:
            list: List of dataset paths for the split
        """
        split_dir = 'valid' if split == 'val' else split
        return [f"{split_dir}/images/{dataset}" for dataset in datasets]


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