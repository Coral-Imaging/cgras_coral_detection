import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional


class DatasetBalancer:
    """
    A class for balancing datasets by ensuring equal numbers of empty and non-empty label files.
    This helps prevent bias in training by equalizing positive and negative examples.
    """

    def __init__(self, root_folder: str, destination_folder: str, empty_threshold: int = 10, max_workers: Optional[int] = None):
        """
        Initialize the DatasetBalancer with source and destination folders.

        Args:
            root_folder: The root directory containing datasets to process
            destination_folder: The directory where balanced dataset will be saved
            empty_threshold: Size in bytes below which a label file is considered empty (default: 10)
            max_workers: Maximum number of threads to use for parallel processing (default: None, uses ThreadPoolExecutor default)
        """
        self.root_folder = root_folder
        self.destination_folder = destination_folder
        self.empty_threshold = empty_threshold
        self.max_workers = max_workers
        
        # Make sure destination folder exists
        os.makedirs(destination_folder, exist_ok=True)

    def find_valid_folders(self) -> List[str]:
        """
        Find folders that contain both 'images' and 'labels' subfolders.
        
        Returns:
            A list of paths to valid folders
        """
        valid_folders = []
        for dirpath, dirnames, _ in os.walk(self.root_folder):
            if 'images' in dirnames and 'labels' in dirnames:
                valid_folders.append(dirpath)
        return valid_folders

    def analyze_labels(self, labels_folder: str) -> Tuple[List[str], List[str]]:
        """
        Analyze label files and separate them into non-empty and empty categories.
        
        Args:
            labels_folder: Path to the folder containing label files
            
        Returns:
            A tuple containing (non_empty_labels, empty_labels)
        """
        # Get lists of label files
        label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
        
        # Separate non-empty and empty label files
        non_empty_labels = [f for f in label_files if os.path.getsize(os.path.join(labels_folder, f)) >= self.empty_threshold]
        empty_labels = [f for f in label_files if os.path.getsize(os.path.join(labels_folder, f)) < self.empty_threshold]
        
        return non_empty_labels, empty_labels

    def process_folder(self, folder: str) -> Tuple[int, int]:
        """
        Process a single folder containing images and labels.
        
        Args:
            folder: Path to the folder containing 'images' and 'labels' subfolders
            
        Returns:
            A tuple containing (number of files processed, number of missing images)
        """
        labels_folder = os.path.join(folder, "labels")
        images_folder = os.path.join(folder, "images")
        
        # Analyze labels
        non_empty_labels, empty_labels = self.analyze_labels(labels_folder)
        
        print(f"Processing {labels_folder}:")
        print("Number of non-empty labels found:", len(non_empty_labels))
        print("Number of empty labels found:", len(empty_labels))
        
        # Ensure equal counts of non-empty and empty labels
        selected_empty_labels = empty_labels[:len(non_empty_labels)]
        selected_labels = non_empty_labels + selected_empty_labels
        print("Total number of labels in new dataset:", len(selected_labels))
        
        missing_images = 0
        files_processed = 0
        
        for label_file in selected_labels:
            label_path = os.path.join(labels_folder, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_folder, image_file)
            
            if os.path.exists(image_path):
                # Define relative path for maintaining structure
                relative_path = os.path.relpath(folder, self.root_folder)
                label_dest = os.path.join(self.destination_folder, relative_path, "labels", label_file)
                image_dest = os.path.join(self.destination_folder, relative_path, "images", image_file)
                
                # Create subdirectories if not exist
                os.makedirs(os.path.dirname(label_dest), exist_ok=True)
                os.makedirs(os.path.dirname(image_dest), exist_ok=True)
                
                # Copy files to destination
                shutil.copy(label_path, label_dest)
                shutil.copy(image_path, image_dest)
                files_processed += 1
            else:
                print(f"Image not found for label: {label_file}")
                missing_images += 1
                
        return files_processed, missing_images

    def balance_dataset(self) -> Tuple[int, int]:
        """
        Process all valid folders to create a balanced dataset.
        
        Returns:
            A tuple containing (total files processed, total missing images)
        """
        valid_folders = self.find_valid_folders()
        
        if not valid_folders:
            print(f"No valid folders found in {self.root_folder}")
            return 0, 0
            
        total_processed = 0
        total_missing = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_folder, folder) for folder in valid_folders]
            
            for future in futures:
                processed, missing = future.result()
                total_processed += processed
                total_missing += missing
        
        print(f"Dataset balancing complete. Processed {total_processed} files. Missing images: {total_missing}")
        return total_processed, total_missing


def main():
    """Command line entry point for the dataset balancer."""
    parser = argparse.ArgumentParser(description='Balance dataset by equalizing empty and non-empty label files.')
    parser.add_argument('--root', type=str, 
                       default="/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/Reduced_dataset_patches/",
                       help='Root folder containing the dataset to balance')
    parser.add_argument('--dest', type=str,
                       default="/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/dataset_2023_built_from_testSet_122/",
                       help='Destination folder for the balanced dataset')
    parser.add_argument('--empty-threshold', type=int, default=10,
                       help='Size in bytes below which a label file is considered empty')
    parser.add_argument('--workers', type=int, default=None,
                       help='Maximum number of worker threads to use')
    
    args = parser.parse_args()
    
    balancer = DatasetBalancer(
        root_folder=args.root,
        destination_folder=args.dest,
        empty_threshold=args.empty_threshold,
        max_workers=args.workers
    )
    
    balancer.balance_dataset()


if __name__ == "__main__":
    main()