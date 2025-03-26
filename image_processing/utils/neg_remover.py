import os
import shutil
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any


class NegRemover:
    """
    A class to balance datasets by ensuring equal numbers of empty (negative) and 
    non-empty (positive) label files.
    """
    
    def __init__(self, root_folder: str, destination_folder: str = None):
        """
        Initialize the NegRemover with source and destination folders.
        
        Args:
            root_folder: The root directory containing image and label data
            destination_folder: Where to store the balanced dataset (defaults to root_folder + '_balanced')
        """
        self.root_folder = root_folder
        
        if destination_folder is None:
            self.destination_folder = f"{root_folder}_balanced"
        else:
            self.destination_folder = destination_folder
        
        self.valid_folders = []
        self.stats = {
            "total_folders": 0,
            "total_non_empty": 0,
            "total_empty": 0,
            "total_selected": 0
        }
    
    def find_valid_folders(self) -> List[str]:
        """
        Find folders that contain both 'images' and 'labels' subfolders.
        
        For datasets with a different structure, we look for a structure where:
        1. There's a direct 'images' and 'labels' folder at the root
        2. Or there are subdirectories containing 'images' and 'labels'
        
        Returns:
            List of valid folders for processing
        """
        valid_folders = []
        
        # Check if the root folder directly contains 'images' and 'labels'
        if (os.path.exists(os.path.join(self.root_folder, 'images')) and 
            os.path.exists(os.path.join(self.root_folder, 'labels'))):
            valid_folders.append(self.root_folder)
        else:
            # Otherwise, look for subdirectories containing 'images' and 'labels'
            for dirpath, dirnames, _ in os.walk(self.root_folder):
                if 'images' in dirnames and 'labels' in dirnames:
                    valid_folders.append(dirpath)
        
        self.valid_folders = valid_folders
        self.stats["total_folders"] = len(valid_folders)
        
        if not valid_folders:
            print(f"Warning: No valid folders containing both 'images' and 'labels' found in {self.root_folder}")
        else:
            print(f"Found {len(valid_folders)} valid folders for processing")
            for folder in valid_folders:
                print(f"  - {folder}")
                
        return valid_folders
    
    def process_folder(self, folder: str) -> Tuple[int, int, int]:
        """
        Process a single folder containing images and labels.
        
        Args:
            folder: Path to a folder containing 'images' and 'labels' subfolders
            
        Returns:
            Tuple of (non_empty_count, empty_count, selected_count)
        """
        labels_folder = os.path.join(folder, "labels")
        images_folder = os.path.join(folder, "images")
        
        # Ensure directories exist
        if not os.path.exists(labels_folder) or not os.path.exists(images_folder):
            print(f"Error: Labels or images folder not found in {folder}")
            return 0, 0, 0
        
        # Get lists of label files
        label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
        
        if not label_files:
            print(f"Warning: No label files found in {labels_folder}")
            return 0, 0, 0
        
        # Separate non-empty and empty label files
        non_empty_labels = []
        empty_labels = []
        
        for label_file in label_files:
            label_path = os.path.join(labels_folder, label_file)
            try:
                if os.path.getsize(label_path) > 0:
                    non_empty_labels.append(label_file)
                else:
                    empty_labels.append(label_file)
            except Exception as e:
                print(f"Error processing {label_path}: {str(e)}")
        
        # If there are no non-empty labels, keep all empty labels
        if not non_empty_labels:
            print(f"Warning: No non-empty labels found in {labels_folder}. Keeping all empty labels.")
            selected_labels = empty_labels
        # If there are no empty labels, keep all non-empty labels
        elif not empty_labels:
            print(f"Warning: No empty labels found in {labels_folder}. Keeping all non-empty labels.")
            selected_labels = non_empty_labels
        # Otherwise, balance the dataset
        else:
            # Ensure equal counts of non-empty and empty labels, randomly selected
            if len(non_empty_labels) <= len(empty_labels):
                # Keep all non-empty and sample from empty
                selected_empty_labels = random.sample(empty_labels, len(non_empty_labels))
                selected_labels = non_empty_labels + selected_empty_labels
            else:
                # Keep all empty and sample from non-empty
                selected_non_empty_labels = random.sample(non_empty_labels, len(empty_labels))
                selected_labels = selected_non_empty_labels + empty_labels
        
        # Create output directories
        relative_path = os.path.relpath(labels_folder, self.root_folder)
        output_labels_dir = os.path.join(self.destination_folder, relative_path)
        output_images_dir = os.path.join(self.destination_folder, os.path.relpath(images_folder, self.root_folder))
        
        os.makedirs(output_labels_dir, exist_ok=True)
        os.makedirs(output_images_dir, exist_ok=True)
        
        # Copy selected files
        copied_count = 0
        
        for label_file in selected_labels:
            label_path = os.path.join(labels_folder, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_folder, image_file)
            
            if os.path.exists(image_path):
                # Define destination paths
                label_dest = os.path.join(output_labels_dir, label_file)
                image_dest = os.path.join(output_images_dir, image_file)
                
                # Copy files to destination
                try:
                    shutil.copy(label_path, label_dest)
                    shutil.copy(image_path, image_dest)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying files for {label_file}: {str(e)}")
            else:
                print(f"Image not found for label: {label_file}")
        
        # Create a Train.txt file with paths to all images in the output directory
        if copied_count > 0:
            self._create_train_txt(output_images_dir, os.path.dirname(output_images_dir))
        
        return len(non_empty_labels), len(empty_labels), copied_count
    
    def _create_train_txt(self, images_dir: str, base_dir: str) -> None:
        """
        Create a Train.txt file with paths to all images.
        
        Args:
            images_dir: Directory containing images
            base_dir: Base directory for relative paths
        """
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            train_txt_path = os.path.join(base_dir, 'Train.txt')
            
            with open(train_txt_path, 'w') as f:
                for image_file in sorted(image_files):
                    # Create path relative to base_dir
                    rel_path = os.path.join(os.path.relpath(images_dir, base_dir), image_file)
                    f.write(f"{rel_path}\n")
            
            # Also create a basic data.yaml file if it doesn't exist
            yaml_path = os.path.join(base_dir, 'data.yaml')
            if not os.path.exists(yaml_path):
                with open(yaml_path, 'w') as f:
                    f.write("names:\n")
                    f.write("  0: alive\n")
                    f.write("  1: dead\n")
                    f.write("  2: mask_live\n")
                    f.write("  3: mask_dead\n")
                    f.write("path: .\n")
                    f.write("Train: Train.txt\n")
    
    def process_all(self, max_workers: int = None, verbose: bool = True) -> Dict[str, Any]:
        """
        Process all valid folders to create a balanced dataset.
        
        Args:
            max_workers: Maximum number of threads to use (None = auto)
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with statistics about the processing
        """
        if not self.valid_folders:
            self.find_valid_folders()
        
        if not self.valid_folders:
            print("No valid folders found. Nothing to process.")
            return self.stats
        
        if verbose:
            print(f"\nProcessing {len(self.valid_folders)} folders to balance negatives/positives...")
        
        total_non_empty = 0
        total_empty = 0
        total_selected = 0
        
        # Process using threading for speedup with I/O bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for folder in self.valid_folders:
                futures.append(executor.submit(self.process_folder, folder))
            
            for i, future in enumerate(futures):
                try:
                    non_empty, empty, selected = future.result()
                    total_non_empty += non_empty
                    total_empty += empty
                    total_selected += selected
                    
                    if verbose and (i+1) % 5 == 0:
                        print(f"  Processed {i+1}/{len(self.valid_folders)} folders...")
                except Exception as e:
                    print(f"Error processing a folder: {str(e)}")
        
        self.stats.update({
            "total_non_empty": total_non_empty,
            "total_empty": total_empty,
            "total_selected": total_selected
        })
        
        if verbose:
            print("\nProcessing complete:")
            print(f"  Total non-empty labels found: {total_non_empty}")
            print(f"  Total empty labels found: {total_empty}")
            print(f"  Total labels in balanced dataset: {total_selected}")
            print(f"  Balanced dataset saved to: {self.destination_folder}")
        
        return self.stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the processed datasets."""
        return self.stats


# Example usage
if __name__ == "__main__":
    # Example usage
    remover = NegRemover(
        root_folder="/home/java/hpc-home/data/train100/export100_from_cvat/exported100",
        destination_folder="/home/java/hpc-home/data/train100/balanced_dataset/exported100"
    )
    
    remover.process_all(verbose=True)