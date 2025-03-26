#!/usr/bin/env python3
import os
import yaml
import shutil
import random
from pathlib import Path
import argparse
from tqdm import tqdm


class DatasetSplitter:
    """
    A class to split YOLO format datasets into train, validation, and test sets.
    
    This splitter handles CVAT-exported YOLO datasets with the specific structure:
    - A main folder containing one or more dataset folders
    - Each dataset folder containing:
      - data/images/Train/*.jpg (images)
      - data/labels/Train/*.txt (labels)
      - data.yaml (class definitions)
      - Train.txt (list of image paths)
    
    The splitter reorganizes the data into train, val, and test folders with matching
    image and label files, and creates a new YAML file for YOLO training.
    """
    
    def __init__(self, input_path, output_path, split_ratios=(0.7, 0.15, 0.15)):
        """
        Initialize the DatasetSplitter.
        
        Args:
            input_path (str): Path to the folder containing the dataset folders.
            output_path (str): Path where the split dataset will be saved.
            split_ratios (tuple): Ratios for train, validation, and test splits. Default: (0.7, 0.15, 0.15)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.split_output_path = self.output_path / "split_dataset"
        self.train_ratio, self.val_ratio, self.test_ratio = split_ratios
        
        # Validate split ratios
        if abs(sum(split_ratios) - 1.0) > 0.001:
            raise ValueError("Split ratios must sum to 1.0")
    
    def process(self):
        """Process all dataset folders in the input path."""
        # Create the split_dataset directory if it doesn't exist
        os.makedirs(self.split_output_path, exist_ok=True)
        
        # List all directories in the input path (each is a dataset)
        dataset_folders = [d for d in self.input_path.iterdir() if d.is_dir()]
        
        if not dataset_folders:
            print(f"No dataset folders found in {self.input_path}")
            return
        
        all_dataset_info = {}
        
        # Process each dataset folder with progress bar
        for dataset_folder in tqdm(dataset_folders, desc="Processing datasets", unit="dataset"):
            print(f"Processing dataset: {dataset_folder.name}")
            dataset_info = self.process_dataset(dataset_folder)
            if dataset_info:
                all_dataset_info[dataset_folder.name] = dataset_info
        
        # Create a master YAML file with information about all split datasets
        self.create_master_yaml(all_dataset_info)
        
        print(f"Dataset splitting complete. Output saved to {self.split_output_path}")
    
    def process_dataset(self, dataset_folder):
        """
        Process a single dataset folder.
        
        Args:
            dataset_folder (Path): Path to the dataset folder.
            
        Returns:
            dict: Information about the processed dataset.
        """
        # Paths to required files and directories
        data_yaml_path = dataset_folder / "data.yaml"
        train_txt_path = dataset_folder / "Train.txt"
        images_path = dataset_folder / "data" / "images" / "Train"
        labels_path = dataset_folder / "data" / "labels" / "Train"
        
        # Check if required files and directories exist
        if not all(p.exists() for p in [data_yaml_path, train_txt_path, images_path, labels_path]):
            print(f"Dataset {dataset_folder.name} is missing required files or directories.")
            return None
        
        # Load class names from data.yaml
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        class_names = data_yaml.get('names', {})
        
        # Create output directory for this dataset
        dataset_output_path = self.split_output_path / dataset_folder.name
        for split in ['train', 'val', 'test']:
            os.makedirs(dataset_output_path / split / 'images', exist_ok=True)
            os.makedirs(dataset_output_path / split / 'labels', exist_ok=True)
        
        # Get all image paths from Train.txt
        with open(train_txt_path, 'r') as f:
            train_paths = [line.strip() for line in f.readlines()]
        
        # Get all actual image files (as a sanity check)
        actual_images = list(images_path.glob('*.jpg'))
        
        # Map from Train.txt paths to actual file paths
        image_files = []
        for rel_path in train_paths:
            # Convert relative path to absolute path within dataset folder
            if rel_path.startswith("data/images/Train/"):
                img_file = dataset_folder / rel_path
                if img_file.exists():
                    image_files.append(img_file)
                else:
                    # Try alternative path construction
                    img_name = os.path.basename(rel_path)
                    alt_img_file = images_path / img_name
                    if alt_img_file.exists():
                        image_files.append(alt_img_file)
                    else:
                        print(f"Warning: Image {rel_path} not found")
            else:
                print(f"Warning: Unexpected path format: {rel_path}")
        
        # If no valid images were found, try using all JPG files in the images directory
        if not image_files:
            print(f"No valid images found from Train.txt, using all JPG files in {images_path}")
            image_files = actual_images
        
        # Shuffle and split the dataset
        random.shuffle(image_files)
        num_samples = len(image_files)
        train_size = int(num_samples * self.train_ratio)
        val_size = int(num_samples * self.val_ratio)
        
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        # Store relative paths for the yaml file
        relative_paths = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # Process each split with progress bars
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            for img_file in tqdm(files, desc=f"Processing {split_name} split", unit="file", leave=False):
                # Get corresponding label file
                img_name = img_file.name
                base_name = os.path.splitext(img_name)[0]
                label_file = labels_path / f"{base_name}.txt"
                
                # New paths
                dest_img_path = dataset_output_path / split_name / 'images' / img_name
                dest_label_path = dataset_output_path / split_name / 'labels' / f"{base_name}.txt"
                
                # Copy image
                shutil.copy2(img_file, dest_img_path)
                
                # Copy label if it exists
                if label_file.exists():
                    shutil.copy2(label_file, dest_label_path)
                else:
                    print(f"Warning: Label not found for {img_name}")
                
                # Add relative path to the list
                rel_path = str(dest_img_path.relative_to(self.split_output_path))
                relative_paths[split_name].append(rel_path)
        
        # Create YAML file for this dataset
        yaml_path = dataset_output_path / f"{dataset_folder.name}_data.yaml"
        yaml_content = {
            'names': class_names,
            'path': str(dataset_output_path.relative_to(self.output_path)),
            'train': relative_paths['train'],
            'val': relative_paths['val'],
            'test': relative_paths['test']
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        print(f"Created dataset YAML: {yaml_path}")
        
        return {
            'class_names': class_names,
            'path': str(dataset_output_path.relative_to(self.split_output_path)),
            'train_count': len(relative_paths['train']),
            'val_count': len(relative_paths['val']),
            'test_count': len(relative_paths['test'])
        }
    
    def create_master_yaml(self, all_dataset_info):
        """
        Create a master YAML file with information about all processed datasets.
        
        Args:
            all_dataset_info (dict): Information about all processed datasets.
        """
        if not all_dataset_info:
            return
        
        # Combine all class names from all datasets
        all_classes = {}
        for info in all_dataset_info.values():
            for class_id, class_name in info['class_names'].items():
                if class_id not in all_classes:
                    all_classes[class_id] = class_name
        
        # Get all train/val/test paths - directories only, not individual files
        all_train_paths = []
        all_val_paths = []
        all_test_paths = []
        
        for dataset_name, info in all_dataset_info.items():
            # Add directory paths instead of individual files
            if info['train_count'] > 0:
                all_train_paths.append(f"{dataset_name}/train/images")
            
            if info['val_count'] > 0:
                all_val_paths.append(f"{dataset_name}/val/images")
            
            if info['test_count'] > 0:
                all_test_paths.append(f"{dataset_name}/test/images")
        
        # Create the master YAML
        master_yaml = {
            'names': all_classes,
            'path': 'split_dataset',
            'train': all_train_paths,
            'val': all_val_paths,
            'test': all_test_paths
        }
        
        master_yaml_path = self.output_path / "cgras_data.yaml"
        with open(master_yaml_path, 'w') as f:
            yaml.dump(master_yaml, f, sort_keys=False)
        
        print(f"Created master YAML: {master_yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Split YOLO datasets into train, validation, and test sets.')
    parser.add_argument('--input', required=True, help='Path to the folder containing dataset folders')
    parser.add_argument('--output', required=True, help='Path where the split dataset will be saved')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio for training set (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio for validation set (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio for test set (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        parser.error(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Create and run the splitter
    splitter = DatasetSplitter(
        input_path=args.input,
        output_path=args.output,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio)
    )
    
    splitter.process()


if __name__ == "__main__":
    main()