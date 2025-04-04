import os
import shutil
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import threading


class FolderStructurer:
    """
    A class to validate, convert, and visualize CVAT export structures for YOLO training.
    """
    
    def __init__(self, input_path=None, output_path=None, max_workers=None):
        """
        Initialize the FolderStructurer with input and output paths.
        
        Args:
            input_path (str): Path to the folder containing CVAT export datasets
            output_path (str): Path where the restructured data should be saved
            max_workers (int, optional): Maximum number of worker threads for parallel processing
        """
        self.input_path = Path(input_path) if input_path else None
        self.output_path = Path(output_path) if output_path else None
        self.yaml_path = None
        self.valid_datasets = []
        self.invalid_datasets = []
        self.validation_errors = {}
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)  # Default from ThreadPoolExecutor
    
    def set_paths(self, input_path, output_path=None, max_workers=None):
        """
        Set or update the input and output paths.
        
        Args:
            input_path (str): Path to the folder containing CVAT export datasets
            output_path (str, optional): Path where the restructured data should be saved
            max_workers (int, optional): Maximum number of worker threads
        """
        self.input_path = Path(input_path)
        if output_path:
            self.output_path = Path(output_path)
        if max_workers:
            self.max_workers = max_workers
    
    def print_expected_structure(self):
        """
        Print the expected folder structure for CVAT exports.
        """
        print("\nExpected folder structure for CVAT exports:")
        print("export_datasets_from_cvat/")
        print("│   ├── dataset1/")
        print("│   │   ├── data/")
        print("│   │   │   ├── labels/")
        print("│   │   │   │   ├── Train/ (containing .txt files)")
        print("│   │   │   ├── images/")
        print("│   │   │   │   ├── Train/ (containing image files)")
        print("│   │   ├── data.yaml (with 'names:', 'path:', and 'train:' fields)")
        print("│   │   ├── Train.txt (with relative paths to image files)")
        print("│   ├── dataset2/...")
        print("\nRequired files and fields:")
        print("1. data.yaml must contain 'names' with class labels, 'path' with relative path to 'Train.txt'")
        print("2. Train.txt must list relative paths to all image files")
        print("3. Image paths must contain '/images/' which can be replaced with '/labels/' to find label files")
        print("4. Both images and label files must exist at the specified locations")
    
    def visualize_input_structure(self):
        """
        Visualize the input folder structure (excluding image and txt files).
        """
        if not self.input_path or not self.input_path.exists():
            print(f"Error: Input path does not exist or is not set.")
            return
        
        print("\nInput Folder Structure:")
        self._visualize_directory(self.input_path, depth=0, max_depth=5)
    
    def visualize_output_structure(self):
        """
        Visualize the output folder structure (excluding image and txt files).
        """
        if not self.output_path or not self.output_path.exists():
            print(f"Error: Output path does not exist or is not set.")
            return
        
        print("\nOutput Folder Structure:")
        self._visualize_directory(self.output_path, depth=0, max_depth=5)
    
    def _visualize_directory(self, path, depth=0, max_depth=3, prefix=""):
        """
        Helper method to recursively visualize a directory structure.
        
        Args:
            path (Path): Path to visualize
            depth (int): Current depth level
            max_depth (int): Maximum depth to visualize
            prefix (str): Prefix for line indentation
        """
        if depth > max_depth:
            print(f"{prefix}└── ...")
            return
        
        # Print current directory
        if depth == 0:
            print(f"{path.name}/")
        
        # Get sorted list of items (directories first)
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        # Skip image and text files
        filtered_items = []
        for item in items:
            # Skip image files and txt files except for Train.txt
            if item.is_file() and (item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
                continue
            if item.is_file() and item.suffix.lower() == '.txt' and item.name.lower() != 'train.txt':
                continue
            filtered_items.append(item)
        
        # Print the items
        for i, item in enumerate(filtered_items):
            is_last = i == len(filtered_items) - 1
            conn = "└── " if is_last else "├── "
            
            # Print current item
            print(f"{prefix}{conn}{item.name}" + ("/" if item.is_dir() else ""))
            
            # Recursively print subdirectories
            if item.is_dir():
                new_prefix = prefix + ("    " if is_last else "│   ")
                self._visualize_directory(item, depth + 1, max_depth, new_prefix)
    
    def _copy_file_task(self, src_path, dst_path, counter_dict, counter_key, counter_lock):
        """
        Task for copying a single file with counter update
        
        Args:
            src_path: Source file path
            dst_path: Destination file path
            counter_dict: Dictionary to store counters
            counter_key: Key for the counter to increment
            counter_lock: Lock for thread-safe counter updates
            
        Returns:
            bool: True if file was copied, False otherwise
        """
        try:
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                with counter_lock:
                    counter_dict[counter_key] += 1
                return True
            else:
                with counter_lock:
                    counter_dict[f"missing_{counter_key}s"] += 1
                return False
        except Exception as e:
            tqdm.write(f"Error copying {src_path}: {str(e)}")
            return False
            
    def _process_file_pair(self, source_dir, rel_img_path, output_images_dir, output_labels_dir, counters, counter_lock, pbar=None):
        """
        Process and copy an image-label file pair
        
        Args:
            source_dir: Source directory root
            rel_img_path: Relative path to image
            output_images_dir: Target directory for images
            output_labels_dir: Target directory for labels
            counters: Dictionary for tracking counts
            counter_lock: Lock for thread-safe counter updates
            pbar: Progress bar to update
            
        Returns:
            None
        """
        # Handle both absolute and relative paths
        if rel_img_path.startswith('/'):
            rel_img_path = rel_img_path[1:]
            
        src_img_path = source_dir / rel_img_path
        
        # Convert image path to label path
        if '/images/' in rel_img_path:
            rel_label_path = rel_img_path.replace('/images/', '/labels/')
            # Ensure label path has .txt extension
            base_path = os.path.splitext(rel_label_path)[0]
            rel_label_path = f"{base_path}.txt"
            src_label_path = source_dir / rel_label_path
        else:
            if pbar:
                pbar.update(2)  # Count both image and label as processed
            return
        
        # Extract just the filename
        img_filename = os.path.basename(rel_img_path)
        label_filename = os.path.basename(rel_label_path)
        
        # Copy files
        self._copy_file_task(src_img_path, output_images_dir / img_filename, counters, "image", counter_lock)
        self._copy_file_task(src_label_path, output_labels_dir / label_filename, counters, "label", counter_lock)
        
        if pbar:
            pbar.update(2)  # Update for both image and label
    
    def validate_cvat_export_structure(self, dataset_folder):
        """
        Validates that a dataset folder follows the expected CVAT export structure.
        
        Args:
            dataset_folder (Path): Path to the dataset folder
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check data.yaml existence
        data_yaml_path = dataset_folder / "data.yaml"
        if not data_yaml_path.exists():
            return False, f"Missing data.yaml file in {dataset_folder}"
        
        # Check data.yaml content
        try:
            with open(data_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
                
            # Validate required fields in data.yaml
            if 'names' not in data_yaml:
                return False, f"Missing 'names' field in data.yaml in {dataset_folder}"
            
            if not isinstance(data_yaml['names'], (list, dict)):
                return False, f"'names' field in data.yaml must be a list or dictionary in {dataset_folder}"
                
            if 'path' not in data_yaml:
                return False, f"Missing 'path' field in data.yaml in {dataset_folder}"
                
            # Get the train file name
            train_file = data_yaml.get('train', 'Train.txt')
                
        except yaml.YAMLError:
            return False, f"Invalid YAML format in data.yaml in {dataset_folder}"
        
        # Check train.txt existence
        train_txt_path = dataset_folder / train_file
        if not train_txt_path.exists():
            return False, f"Missing {train_file} file in {dataset_folder}"
        
        # Check if train.txt has content
        try:
            with open(train_txt_path, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
                
            if not image_paths:
                return False, f"{train_file} is empty in {dataset_folder}"
                
            # Validate some image paths
            source_dir = dataset_folder
            images_exist = False
            labels_exist = False
            
            for rel_img_path in image_paths[:5]:  # Check first 5 images for validation
                if rel_img_path.startswith('/'):
                    rel_img_path = rel_img_path[1:]
                    
                src_img_path = source_dir / rel_img_path
                if src_img_path.exists():
                    images_exist = True
                    
                # Check if path contains '/images/' which can be replaced with '/labels/'
                if '/images/' not in rel_img_path:
                    return False, f"Image path doesn't contain '/images/' directory: {rel_img_path} in {dataset_folder}"
                    
                # Change this part to handle .txt extension
                rel_label_path = rel_img_path.replace('/images/', '/labels/')
                # Ensure label path has .txt extension
                base_path = os.path.splitext(rel_label_path)[0]
                rel_label_path = f"{base_path}.txt"
                src_label_path = source_dir / rel_label_path
                
                if src_label_path.exists():
                    labels_exist = True
                    
            if not images_exist:
                return False, f"No images found at paths listed in {train_file} in {dataset_folder}"
                
            if not labels_exist:
                return False, f"No label files found at expected locations in {dataset_folder}"
                
        except Exception as e:
            return False, f"Error reading {train_file} in {dataset_folder}: {str(e)}"
        
        return True, ""
    
    def validate_input(self):
        """
        Validate the entire input folder structure.
        
        Returns:
            bool: True if all datasets are valid or no errors were found
        """
        if not self.input_path:
            print("Error: Input path not set.")
            return False
            
        if not self.input_path.exists() or not self.input_path.is_dir():
            print(f"Error: Input path '{self.input_path}' does not exist or is not a directory")
            self.print_expected_structure()
            return False
            
        # Check if there are any directories in the input path
        datasets = [d for d in self.input_path.iterdir() if d.is_dir()]
        if not datasets:
            print(f"Error: No dataset directories found in '{self.input_path}'")
            self.print_expected_structure()
            return False
        
        # Reset validation results
        self.valid_datasets = []
        self.invalid_datasets = []
        self.validation_errors = {}
        
        # Validate each dataset
        for dataset_folder in datasets:
            dataset_name = dataset_folder.name
            print(f"Validating dataset: {dataset_name}")
            
            is_valid, error_message = self.validate_cvat_export_structure(dataset_folder)
            if is_valid:
                self.valid_datasets.append(dataset_folder)
                print(f"✓ {dataset_name} structure is valid")
            else:
                self.invalid_datasets.append(dataset_folder)
                self.validation_errors[dataset_name] = error_message
                print(f"✗ {dataset_name} validation failed: {error_message}")
        
        # Summary
        print(f"\nValidation summary:")
        print(f"  - Valid datasets: {len(self.valid_datasets)}")
        print(f"  - Invalid datasets: {len(self.invalid_datasets)}")
        
        if self.invalid_datasets:
            print("\nThe following datasets have validation errors:")
            for dataset in self.invalid_datasets:
                print(f"  - {dataset.name}: {self.validation_errors[dataset.name]}")
            return False
        
        return True
    
    def convert(self, ignore_validation=False):
        """
        Convert the validated datasets to the YOLO format.
        
        Args:
            ignore_validation (bool): If True, attempt to convert all datasets regardless of validation
            
        Returns:
            bool: True if conversion was successful
        """
        if not self.input_path or not self.output_path:
            print("Error: Input and output paths must be set before conversion.")
            return False
            
        if not ignore_validation and not self.valid_datasets and not self.validate_input():
            print("Conversion aborted due to validation errors.")
            print("Run validate_input() first or use ignore_validation=True to force conversion.")
            return False
        
        # Initialize variables to store class names and dataset paths
        class_idx_to_name = {}  # Preserve the class index to name mapping
        highest_idx = -1        # Track the highest class index
        dataset_paths = []
        
        # Process each valid dataset
        datasets_to_process = self.input_path.iterdir() if ignore_validation else self.valid_datasets
        datasets_to_process = [d for d in datasets_to_process if d.is_dir()]
        
        print(f"\nConverting {len(datasets_to_process)} datasets...")
        dataset_pbar = tqdm(datasets_to_process, desc="Overall Progress", unit="dataset")
        
        # First pass: collect and merge class information with preserved indices
        for dataset_folder in datasets_to_process:
            dataset_name = dataset_folder.name
            
            try:
                data_yaml_path = dataset_folder / "data.yaml"
                if not data_yaml_path.exists():
                    print(f"Warning: data.yaml not found in {dataset_name}, skipping class analysis...")
                    continue
                    
                with open(data_yaml_path, 'r') as f:
                    data_yaml = yaml.safe_load(f)
                    
                if 'names' in data_yaml:
                    names = data_yaml.get('names', {})
                    
                    # Handle dictionary format (most common in your case)
                    if isinstance(names, dict):
                        for idx_str, name in names.items():
                            # Convert string indices to integers
                            idx = int(idx_str) if isinstance(idx_str, str) and idx_str.isdigit() else idx_str
                            
                            # Only add if this exact index isn't already mapped
                            if idx not in class_idx_to_name:
                                class_idx_to_name[idx] = name
                                highest_idx = max(highest_idx, idx)
                    
                    # Handle list format (convert to dict with indices)
                    elif isinstance(names, list):
                        for i, name in enumerate(names):
                            # Find a suitable index if there's a conflict
                            idx = i
                            while idx in class_idx_to_name and class_idx_to_name[idx] != name:
                                idx = highest_idx + 1
                            
                            if idx not in class_idx_to_name:
                                class_idx_to_name[idx] = name
                                highest_idx = max(highest_idx, idx)
                
                print(f"Collected classes from {dataset_name}: {class_idx_to_name}")
                
            except Exception as e:
                print(f"Error analyzing classes in {dataset_name}: {str(e)}")
        

        # Create global output directories just once
        output_images_dir = self.output_path / "data" / "images"
        output_labels_dir = self.output_path / "data" / "labels"
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        # Second pass: actual dataset conversion
        for dataset_folder in dataset_pbar:
            dataset_name = dataset_folder.name
            dataset_pbar.set_description(f"Processing dataset: {dataset_name}")
            
            # Get file paths and setup
            try:
                data_yaml_path = dataset_folder / "data.yaml"
                if not data_yaml_path.exists():
                    print(f"Warning: data.yaml not found in {dataset_name}, skipping...")
                    continue
                    
                with open(data_yaml_path, 'r') as f:
                    data_yaml = yaml.safe_load(f)
                    
                # Get the train.txt file path
                train_txt_name = data_yaml.get('train', 'Train.txt')
                train_txt_path = dataset_folder / train_txt_name
                
                if not train_txt_path.exists():
                    print(f"Warning: {train_txt_name} not found in {dataset_name}, skipping...")
                    continue
                    
                # Read the train.txt file to get image paths
                with open(train_txt_path, 'r') as f:
                    image_paths = [line.strip() for line in f.readlines()]
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                continue
                
            # Track statistics for reporting
            counters = {
                "image": 0,
                "label": 0,
                "missing_images": 0,
                "missing_labels": 0
            }
            counter_lock = threading.Lock()  # Lock for thread-safe counter updates
            
            # Create progress bar for this dataset
            total_files = len(image_paths) * 2  # Each image path has a potential label
            pbar = tqdm(total=total_files, 
                        desc=f"Processing {dataset_name}", 
                        unit="files",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
                
            # Copy images and labels to the output directory in parallel
            source_dir = dataset_folder
            
            # Process files in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for rel_img_path in image_paths:
                    future = executor.submit(
                        self._process_file_pair,
                        source_dir,
                        rel_img_path,
                        output_images_dir,
                        output_labels_dir,
                        counters,
                        counter_lock,
                        pbar
                    )
                    futures.append(future)
                
                # Wait for all tasks to complete
                concurrent.futures.wait(futures)
            
            # Close progress bar
            pbar.close()
            
            # Copy counter values to standard variable names for consistency with the rest of the code
            images_copied = counters["image"]
            labels_copied = counters["label"]
            missing_images = counters["missing_images"]
            missing_labels = counters["missing_labels"]
                    
            # Add this dataset to our list for the combined YAML
            dataset_paths.append(dataset_name)
            
            # Print stats for this dataset (only show missing files to reduce output noise)
            stats_msg = f"Dataset {dataset_name}: {images_copied} images, {labels_copied} labels"
            if missing_images > 0 or missing_labels > 0:
                stats_msg += f" (Missing: {missing_images} images, {missing_labels} labels)"
            tqdm.write(stats_msg)
        
        cgras_yaml = {
            'names': class_idx_to_name,  # Use the preserved class mapping
            'path': str(self.output_path.absolute()),
            'data': ['data/images']
        }
        
        # Write the combined YAML file
        with open(self.output_path / "cgras_data.yaml", 'w') as f:
            yaml.dump(cgras_yaml, f, default_flow_style=False, sort_keys=False)
            self.yaml_path = self.output_path / "cgras_data.yaml"
            
        # Close the overall progress bar if it exists
        if 'dataset_pbar' in locals():
            dataset_pbar.close()
        
        print(f"\nSuccessfully created training structure at {self.output_path}")
        print(f"Created cgras_data.yaml with {len(class_idx_to_name)} classes and {len(dataset_paths)} datasets")
        print(f"Class mapping preserved: {class_idx_to_name}")
        
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CVAT exports to YOLO training format")
    parser.add_argument("input_path", help="Path to the folder containing CVAT export datasets")
    parser.add_argument("output_path", help="Path where the restructured data should be saved")
    parser.add_argument("--force", action="store_true", help="Skip validation and force conversion")
    parser.add_argument("--visualize", action="store_true", help="Visualize input and output folder structures")
    parser.add_argument("--threads", type=int, help="Number of worker threads (default: CPU count + 4)")
    
    args = parser.parse_args()
    
    structurer = FolderStructurer(args.input_path, args.output_path, args.threads)
    
    if args.visualize:
        structurer.visualize_input_structure()
    
    if args.force:
        print("Warning: Skipping validation due to --force flag")
        success = structurer.convert(ignore_validation=True)
    else:
        is_valid = structurer.validate_input()
        if not is_valid:
            user_input = input("\nSome datasets have validation errors. Do you want to continue with valid datasets only? (y/n): ")
            if user_input.lower() != 'y':
                print("Conversion aborted. Please fix the errors and try again.")
                exit(1)
        
        success = structurer.convert()
    
    if success and args.visualize:
        structurer.visualize_output_structure()