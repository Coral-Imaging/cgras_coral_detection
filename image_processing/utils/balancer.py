import os
import yaml
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Any, Optional, Union


class DatasetBalancer:
    """
    A class to balance datasets by ensuring equal numbers of empty (negative) and 
    non-empty (positive) label files within each dataset.
    
    This follows the same pattern as other CGRAS data processing tools and operates
    on YAML configuration files.
    """
    
    def __init__(self, yaml_path, output_path=None, max_workers=None):
        """
        Initialize the Dataset Balancer with paths.
        
        Args:
            yaml_path (str): Path to the YOLO data YAML file
            output_path (str, optional): Path where balanced data should be saved
            max_workers (int, optional): Maximum number of worker threads to use
        """
        self.yaml_path = Path(yaml_path)
        self.output_path = Path(output_path) if output_path else self.yaml_path.parent / "balanced_data"
        self.yaml_data = None
        self.max_workers = max_workers if max_workers else min(32, os.cpu_count() + 4)
        self.dataset_stats = {}
        self.new_yaml_path = None
        
        # Load YAML data
        self._load_yaml()
    
    def _load_yaml(self):
        """Load the YAML configuration file"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found at {self.yaml_path}")
            
        with open(self.yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
                
        # Validate YAML has required fields
        if 'names' not in self.yaml_data:
            raise ValueError("Missing 'names' field in YAML file")
                
        # Determine base directory (where the YAML file is located)
        self.base_dir = self.yaml_path.parent
            
        # Find data paths from all potential sources
        self.data_paths = []
        self.dataset_paths = {}  # Store paths by dataset type (train/val/test)
        
        # Check all possible dataset path fields: 'data', 'train', 'val', 'test'
        for key in ['data', 'train', 'val', 'test']:
            if key in self.yaml_data:
                paths = []
                
                # List of paths
                if isinstance(self.yaml_data[key], list):
                    paths.extend(self.yaml_data[key])
                # Single string path
                elif isinstance(self.yaml_data[key], str):
                    paths.append(self.yaml_data[key])
                # Dictionary with paths
                elif isinstance(self.yaml_data[key], dict):
                    for dataset_name, dataset_info in self.yaml_data[key].items():
                        if isinstance(dataset_info, dict) and 'images' in dataset_info:
                            paths.append(dataset_info['images'])
                
                self.dataset_paths[key] = paths
                self.data_paths.extend(paths)
                    
        if not self.data_paths:
            raise ValueError("No dataset paths found in YAML file")
                
        print(f"Found {len(self.data_paths)} dataset paths in the YAML file")
    
    def _find_label_path(self, image_path):
        """
        For a given image path, determine the corresponding label path.
        
        Args:
            image_path (Path): Path to an image
            
        Returns:
            Path: Path to the corresponding label file
        """
        # Convert image path to label path
        # Usually by replacing 'images' with 'labels' and changing extension to .txt
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
    
    def analyze_dataset_balance(self):
        """
        Analyze all datasets to determine the balance between empty and non-empty label files,
        grouped by data split (train/val/test).
        
        Returns:
            dict: Dictionary containing statistics for each split type
        """
        self.dataset_stats = {}
        self.split_stats = {}  # New structure to track stats per split
        dataset_totals = {'non_empty': 0, 'empty': 0, 'total': 0}
        
        # Initialize split stats for each available split type
        for split_type in self.dataset_paths.keys():
            if split_type != 'data':  # Skip the generic 'data' key
                self.split_stats[split_type] = {
                    'non_empty': 0, 
                    'empty': 0, 
                    'total': 0,
                    'datasets': [],
                    'balanced_sample': 0
                }
        
        # Process each dataset path
        for data_path in self.data_paths:
            # Resolve the full path
            full_path = self.base_dir / data_path
            
            # Extract dataset name from path
            dataset_name = self._extract_dataset_name(data_path)
            
            # Skip if dataset_name is already processed (duplicate path in different sections)
            if dataset_name in self.dataset_stats:
                continue
            
            # Determine which split this dataset belongs to
            split_type = None
            for key, paths in self.dataset_paths.items():
                if data_path in paths:
                    split_type = key
                    break
            
            if not split_type:
                split_type = 'data'  # Default if not found
                
            print(f"Analyzing dataset balance in {dataset_name} (split: {split_type})...")
            
            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(full_path.glob(f"**/*{ext}")))
                
            if not image_files:
                print(f"No image files found in {full_path}")
                continue
            
            # Initialize stats for this dataset
            self.dataset_stats[dataset_name] = {
                'path': data_path,
                'split_type': split_type,
                'non_empty': 0,
                'empty': 0,
                'total': len(image_files),
                'balanced_sample': 0
            }
            
            # Using ThreadPoolExecutor to analyze labels in parallel
            with tqdm(total=len(image_files), desc=f"Analyzing {dataset_name}", unit="files") as pbar:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Define a function to check if a label is empty
                    def check_label(img_path):
                        label_path = self._find_label_path(img_path)
                        if label_path.exists():
                            try:
                                return 'non_empty' if os.path.getsize(label_path) > 0 else 'empty'
                            except Exception:
                                return 'error'
                        return 'missing'
                    
                    # Submit all tasks and collect results
                    future_to_path = {executor.submit(check_label, img_path): img_path for img_path in image_files}
                    
                    for future in concurrent.futures.as_completed(future_to_path):
                        result = future.result()
                        if result == 'non_empty':
                            self.dataset_stats[dataset_name]['non_empty'] += 1
                        elif result == 'empty':
                            self.dataset_stats[dataset_name]['empty'] += 1
                        # No need to add anything for 'missing' or 'error'
                        pbar.update(1)
            
            # Calculate the balanced sample size for this dataset
            non_empty = self.dataset_stats[dataset_name]['non_empty']
            empty = self.dataset_stats[dataset_name]['empty']
            self.dataset_stats[dataset_name]['balanced_sample'] = min(non_empty, empty) * 2 if non_empty > 0 and empty > 0 else max(non_empty, empty)
            
            # Update dataset totals
            dataset_totals['non_empty'] += non_empty
            dataset_totals['empty'] += empty
            dataset_totals['total'] += len(image_files)
            
            # Update split stats if this belongs to a specific split
            if split_type in self.split_stats:
                self.split_stats[split_type]['non_empty'] += non_empty
                self.split_stats[split_type]['empty'] += empty
                self.split_stats[split_type]['total'] += len(image_files)
                self.split_stats[split_type]['datasets'].append(dataset_name)
            
            # Print summary for this dataset
            print(f"  - Dataset: {dataset_name} (split: {split_type})")
            print(f"    - Total images: {len(image_files)}")
            print(f"    - Non-empty labels: {non_empty} ({non_empty/len(image_files)*100:.1f}%)")
            print(f"    - Empty labels: {empty} ({empty/len(image_files)*100:.1f}%)")
            print(f"    - Balanced sample size: {self.dataset_stats[dataset_name]['balanced_sample']}")
        
        # Calculate balanced sample size for each split
        for split_type, stats in self.split_stats.items():
            if stats['non_empty'] > 0 and stats['empty'] > 0:
                stats['balanced_sample'] = min(stats['non_empty'], stats['empty']) * 2
            else:
                stats['balanced_sample'] = max(stats['non_empty'], stats['empty'])
        
        # Calculate overall balance
        self.dataset_stats['__overall__'] = dataset_totals
        self.dataset_stats['__overall__']['balanced_sample'] = sum(
            stats['balanced_sample'] for name, stats in self.dataset_stats.items() if name != '__overall__'
        )
        
        # Print summary by split type
        print("\nSummary by Split Type:")
        for split_type, stats in self.split_stats.items():
            # Calculate percentages correctly with proper checks
            if stats['total'] > 0:
                non_empty_pct = stats['non_empty'] / stats['total'] * 100
                empty_pct = stats['empty'] / stats['total'] * 100
            else:
                non_empty_pct = 0
                empty_pct = 0
                
            print(f"  - Split: {split_type}")
            print(f"    - Total images: {stats['total']}")
            print(f"    - Non-empty labels: {stats['non_empty']} ({non_empty_pct:.1f}%)")
            print(f"    - Empty labels: {stats['empty']} ({empty_pct:.1f}%)")
            print(f"    - Balanced sample size: {stats['balanced_sample']}")

        # Print overall summary
        print("\nOverall Summary:")
        # Calculate overall percentages with proper checks
        if dataset_totals['total'] > 0:
            overall_non_empty_pct = dataset_totals['non_empty'] / dataset_totals['total'] * 100
            overall_empty_pct = dataset_totals['empty'] / dataset_totals['total'] * 100
        else:
            overall_non_empty_pct = 0
            overall_empty_pct = 0
            
        print(f"  - Total images: {dataset_totals['total']}")
        print(f"  - Non-empty labels: {dataset_totals['non_empty']} ({overall_non_empty_pct:.1f}%)")
        print(f"  - Empty labels: {dataset_totals['empty']} ({overall_empty_pct:.1f}%)")
        print(f"  - Total balanced sample size: {self.dataset_stats['__overall__']['balanced_sample']}")
        
        return self.dataset_stats
    
    def _extract_dataset_name(self, path):
        """
        Extract a dataset name from a path like 'train/dataset1/images'
        
        Args:
            path (str): Path string
            
        Returns:
            str: Extracted dataset name
        """
        parts = path.split('/')
        dataset_name = None
        
        # Try to find a part before 'images'
        for i, part in enumerate(parts):
            if i < len(parts) - 1 and parts[i+1] == 'images':
                dataset_name = part
                break
        
        # If not found, try to extract from the path structure
        if dataset_name is None:
            if len(parts) >= 2:
                dataset_name = parts[-2]  # Assume second-to-last part is the dataset name
            else:
                dataset_name = parts[0]  # Fallback to first part
        
        return dataset_name
    
    def plot_dataset_balance(self):
        """
        Plot the balance between empty and non-empty labels for all splits (train/val/test).
        
        Returns:
            None
        """
        if not self.dataset_stats:
            print("No dataset statistics available. Running analysis first...")
            self.analyze_dataset_balance()
        
        if not hasattr(self, 'split_stats'):
            print("No split statistics available. Running analysis first...")
            self.analyze_dataset_balance()
        
        # 1. Plot counts by dataset grouped by split type
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for plotting by split
        splits = list(self.split_stats.keys())
        
        # Set up the plot
        split_positions = {}
        current_pos = 0
        
        # Keep track of actual plotted splits and their colors
        plotted_splits = []
        split_colors = {}
        
        # Plot each split's datasets
        for i, split in enumerate(splits):
            datasets = self.split_stats[split]['datasets']
            if not datasets:
                continue
                
            # Track dataset counts for this split
            non_empty_counts = []
            empty_counts = []
            dataset_names = []
            
            # Collect data
            for ds_name in datasets:
                if ds_name in self.dataset_stats:
                    dataset_names.append(ds_name)
                    non_empty_counts.append(self.dataset_stats[ds_name]['non_empty'])
                    empty_counts.append(self.dataset_stats[ds_name]['empty'])
            
            # If no valid datasets with data, skip this split
            if not dataset_names:
                continue
                
            # Calculate positions for this split
            n_datasets = len(dataset_names)
            positions = np.arange(current_pos, current_pos + n_datasets)
            split_positions[split] = positions
            
            # Plot non-empty and empty counts for this split
            width = 0.35
            color_idx = len(plotted_splits)  # Use actual plotted count for color
            
            # Plot bars with consistent colors
            bar1 = ax.bar(positions - width/2, non_empty_counts, width, 
                label=f'{split} - Non-Empty' if split not in plotted_splits else '', 
                color=f'C{color_idx}', alpha=0.8)
            bar2 = ax.bar(positions + width/2, empty_counts, width, 
                label=f'{split} - Empty' if split not in plotted_splits else '', 
                color=f'C{color_idx}', alpha=0.4)
            
            # Store color information for legend
            split_colors[split] = (bar1[0], color_idx)
            
            # Add to plotted splits
            if split not in plotted_splits:
                plotted_splits.append(split)
            
            # Add dataset names
            ax.set_xticks(positions)
            ax.set_xticklabels(dataset_names, rotation=45, ha='right')
            
            # Add split separator and label
            if i < len(splits) - 1 and n_datasets > 0:
                ax.axvline(x=current_pos + n_datasets - 0.5, color='black', linestyle='--', alpha=0.3)
                
            # Update position counter
            current_pos += n_datasets + 1  # +1 for spacing between splits
        
        # Set plot labels
        ax.set_title('Non-Empty vs Empty Labels by Dataset (Grouped by Split)')
        ax.set_ylabel('Number of Labels')
        ax.set_xlabel('Datasets')
        
        # Create custom legend with one entry per actually plotted split
        custom_legend = []
        for split in plotted_splits:
            if split in split_positions and len(split_positions[split]) > 0:
                # Use the stored bar and color information
                bar, _ = split_colors[split]
                custom_legend.append((bar, split))
        
        # Add the legend if we have any items
        if custom_legend:
            ax.legend(*zip(*custom_legend), loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Plot original vs balanced by split type
        plt.figure(figsize=(10, 6))
        
        # Prepare data - only include splits that have data
        split_names = []
        original_counts = []
        balanced_counts = []
        
        for split in self.split_stats.keys():
            if self.split_stats[split]['total'] > 0:
                split_names.append(split)
                original_counts.append(self.split_stats[split]['total'])
                balanced_counts.append(self.split_stats[split]['balanced_sample'])
        
        # Skip plotting if no data
        if not split_names:
            print("No splits with data to plot.")
            return
        
        # Plot
        x = np.arange(len(split_names))
        width = 0.35
        
        plt.bar(x - width/2, original_counts, width, label='Original')
        plt.bar(x + width/2, balanced_counts, width, label='Balanced')
        
        plt.title('Original vs Balanced Dataset Sizes by Split Type')
        plt.xlabel('Split Type')
        plt.ylabel('Number of Images')
        plt.xticks(x, split_names)
        
        # Add counts on bars
        for i, v in enumerate(original_counts):
            plt.text(i - width/2, v + 0.1, str(v), ha='center', va='bottom')
        for i, v in enumerate(balanced_counts):
            plt.text(i + width/2, v + 0.1, str(v), ha='center', va='bottom')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def balance_datasets(self, random_seed=42):
        """
        Balance all datasets by ensuring equal numbers of empty and non-empty label files,
        treating each split (train/val/test) as a whole group.
        
        Args:
            random_seed (int): Random seed for reproducibility
            
        Returns:
            dict: Statistics about the balancing process
        """
        if not self.dataset_stats:
            print("No dataset statistics available. Running analysis first...")
            self.analyze_dataset_balance()
        
        if not hasattr(self, 'split_stats'):
            print("Split statistics not found. Running analysis first...")
            self.analyze_dataset_balance()
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Track balanced stats
        balanced_stats = {}
        
        # Process each split type
        for split_type, split_stats in self.split_stats.items():
            print(f"\nBalancing split: {split_type}")
            
            # Skip if no datasets in this split
            if not split_stats['datasets']:
                print(f"No datasets found for split: {split_type}")
                continue
            
            # Gather all images for this split
            non_empty_images = []
            empty_images = []
            
            # Process each dataset in this split
            for dataset_name in split_stats['datasets']:
                # Get dataset info
                data_path = self.dataset_stats[dataset_name]['path']
                full_path = self.base_dir / data_path
                
                # Find all image files
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.extend(list(full_path.glob(f"**/*{ext}")))
                
                if not image_files:
                    print(f"No image files found in {full_path}")
                    continue
                
                print(f"Categorizing images for dataset: {dataset_name}...")
                with tqdm(total=len(image_files), desc="Categorizing", unit="files") as pbar:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # Define a function to check if a label is empty
                        def check_label_and_categorize(img_path):
                            label_path = self._find_label_path(img_path)
                            if label_path.exists():
                                try:
                                    if os.path.getsize(label_path) > 0:
                                        return ('non_empty', img_path, label_path)
                                    else:
                                        return ('empty', img_path, label_path)
                                except Exception:
                                    return ('error', img_path, None)
                            return ('missing', img_path, None)
                        
                        # Submit all tasks and collect results
                        futures = [executor.submit(check_label_and_categorize, img_path) for img_path in image_files]
                        
                        for future in concurrent.futures.as_completed(futures):
                            category, img_path, label_path = future.result()
                            if category == 'non_empty':
                                non_empty_images.append((img_path, label_path, dataset_name))
                            elif category == 'empty':
                                empty_images.append((img_path, label_path, dataset_name))
                            pbar.update(1)
            
            # Determine how to balance the split as a whole
            non_empty_count = len(non_empty_images)
            empty_count = len(empty_images)
            
            print(f"Split {split_type} has {non_empty_count} non-empty and {empty_count} empty labels")
            
            if non_empty_count == 0:
                print(f"Warning: No non-empty labels found in {split_type}. Skipping.")
                continue
                
            if empty_count == 0:
                print(f"Warning: No empty labels found in {split_type}. Keeping all non-empty labels.")
                selected_non_empty = non_empty_images
                selected_empty = []
            elif non_empty_count <= empty_count:
                # Keep all non-empty and sample from empty
                selected_non_empty = non_empty_images
                selected_empty = random.sample(empty_images, non_empty_count)
            else:
                # Keep all empty and sample from non-empty
                selected_non_empty = random.sample(non_empty_images, empty_count)
                selected_empty = empty_images
            
            # Copy selected files, preserving their original relative paths
            selected_images = selected_non_empty + selected_empty
            copied_count = 0
            
            print(f"Copying {len(selected_images)} files to balanced dataset...")
            with tqdm(total=len(selected_images), desc="Copying", unit="files") as pbar:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    copy_lock = threading.Lock()
                    
                    def copy_files(img_path, label_path, dataset_name):
                        try:
                            # Get dataset info
                            data_path = self.dataset_stats[dataset_name]['path']
                            full_path = self.base_dir / data_path
                            
                            # Get relative paths from the dataset directory
                            rel_img_path = img_path.relative_to(full_path.parent)
                            
                            # Determine destination paths
                            dst_img_path = self.output_path / rel_img_path
                            dst_label_path = self._find_label_path(dst_img_path)
                            
                            # Create directory structure
                            os.makedirs(dst_img_path.parent, exist_ok=True)
                            os.makedirs(dst_label_path.parent, exist_ok=True)
                            
                            # Copy files
                            shutil.copy2(img_path, dst_img_path)
                            shutil.copy2(label_path, dst_label_path)
                            
                            with copy_lock:
                                return True
                        except Exception as e:
                            print(f"Error copying files: {str(e)}")
                            return False
                    
                    # Submit all copy tasks
                    futures = [executor.submit(copy_files, img_path, label_path, dataset_name) 
                            for img_path, label_path, dataset_name in selected_images]
                    
                    for future in concurrent.futures.as_completed(futures):
                        if future.result():
                            copied_count += 1
                        pbar.update(1)
            
            # Update stats
            balanced_stats[split_type] = {
                'original_non_empty': non_empty_count,
                'original_empty': empty_count,
                'original_total': non_empty_count + empty_count,
                'balanced_non_empty': len(selected_non_empty),
                'balanced_empty': len(selected_empty),
                'balanced_total': len(selected_non_empty) + len(selected_empty),
                'copied': copied_count
            }
            
            print(f"Split {split_type} balanced:")
            print(f"  - Original: {non_empty_count} non-empty, {empty_count} empty ({non_empty_count + empty_count} total)")
            print(f"  - Balanced: {len(selected_non_empty)} non-empty, {len(selected_empty)} empty ({len(selected_non_empty) + len(selected_empty)} total)")
            print(f"  - Files copied: {copied_count}")
        
        # Generate new YAML file
        self._generate_yaml()
        
        # Print overall summary
        total_original = sum(stats['original_total'] for stats in balanced_stats.values())
        total_balanced = sum(stats['balanced_total'] for stats in balanced_stats.values())
        total_copied = sum(stats['copied'] for stats in balanced_stats.values())
        
        print("\nBalancing complete:")
        print(f"  - Original dataset total: {total_original} images")
        print(f"  - Balanced dataset total: {total_balanced} images")
        print(f"  - Files successfully copied: {total_copied}")
        print(f"  - Balanced dataset saved to: {self.output_path}")
        print(f"  - YAML file saved to: {self.new_yaml_path}")
        
        return balanced_stats
    
    def _create_output_structure(self, data_path):
        """
        Create output directory structure for a dataset.
        
        Args:
            data_path (str): Path to dataset
            
        Returns:
            Path: Output directory path
        """
        # Determine dataset type (train/val/test)
        dataset_type = None
        for key, paths in self.dataset_paths.items():
            if data_path in paths:
                dataset_type = key
                break
        
        if not dataset_type:
            dataset_type = 'data'  # Default if not found
        
        # Create the output structure
        dataset_name = self._extract_dataset_name(data_path)
        
        # Keep the same structure as the original
        rel_path = Path(data_path).parent
        output_path = self.output_path / rel_path
        
        # Create the directory
        os.makedirs(output_path, exist_ok=True)
        
        return output_path
    
    def _generate_yaml(self):
        """
        Generate a new YAML file for the balanced dataset.
        
        Returns:
            None
        """
        # Copy the original YAML data
        balanced_yaml = self.yaml_data.copy()
        
        # Update the 'path' field
        balanced_yaml['path'] = str(self.output_path.absolute())
        
        # Update dataset paths
        for key, paths in self.dataset_paths.items():
            if key in balanced_yaml:
                # For list format
                if isinstance(balanced_yaml[key], list):
                    balanced_yaml[key] = paths  # Keep the same relative paths
                # For string format
                elif isinstance(balanced_yaml[key], str):
                    balanced_yaml[key] = paths[0] if paths else balanced_yaml[key]
                # For dict format
                elif isinstance(balanced_yaml[key], dict):
                    # Keep the same structure, paths will be relative to new location
                    pass
        
        # Write the YAML file
        yaml_path = self.output_path / "balanced_data.yaml"
        self.new_yaml_path = yaml_path
        with open(yaml_path, 'w') as f:
            yaml.dump(balanced_yaml, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Balance YOLO datasets by equalizing empty and non-empty labels")
    parser.add_argument("yaml_path", help="Path to the YOLO data YAML file")
    parser.add_argument("--output", help="Path where balanced data should be saved")
    parser.add_argument("--analyze", action="store_true", help="Only analyze label balance without creating balanced dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--threads", type=int, help="Number of worker threads")
    
    args = parser.parse_args()
    
    balancer = DatasetBalancer(args.yaml_path, args.output, args.threads)
    
    if args.analyze:
        balancer.analyze_dataset_balance()
        balancer.plot_dataset_balance()
    else:
        print("Analyzing dataset balance...")
        balancer.analyze_dataset_balance()
        balancer.plot_dataset_balance()
        
        print("\nProceed with balancing using these settings? (y/n)")
        user_input = input()
        if user_input.lower() == 'y':
            balancer.balance_datasets(random_seed=args.seed)
        else:
            print("Balancing canceled.")