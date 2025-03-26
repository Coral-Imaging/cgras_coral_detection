import os
import shutil
import glob
import random

class DatasetSplitter:
    def __init__(self, data_location, save_dir, 
             train_ratio=0.70, test_ratio=0.15, valid_ratio=0.15):
        """
        Initialize the DatasetSplitter class.
        
        Args:
            data_path: Base path for the dataset
            data_location: Path to the data directory
            save_dir: Path to save the split dataset
            train_ratio: Ratio of training data
            test_ratio: Ratio of test data
            valid_ratio: Ratio of validation data
            use_direct_paths: Whether to use direct paths instead of joining with data_path
        """
        self.data_location = data_location
        self.save_dir = save_dir
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        
        self._validate_ratios()
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Find all image and label files
        self._find_files()
        self.total_files = len(self.label_list)

    def _validate_ratios(self):
        """Validate that the ratios are valid."""
        if not (0 <= self.test_ratio <= 1 and 0 <= self.train_ratio <= 1 and 0 <= self.valid_ratio <= 1):
            raise ValueError("All ratios must be between 0 and 1.")
        if not (self.train_ratio + self.test_ratio + self.valid_ratio == 1):
            raise ValueError("The sum of train, validation, and test ratios must equal 1.")
    
    def _find_files(self):
        """Find all image and label files in the dataset."""
        # Check if the data location has the expected structure
        images_dir = os.path.join(self.data_location, 'images')
        labels_dir = os.path.join(self.data_location, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Expected directory structure not found at {self.data_location}. "
                                   f"Make sure 'images' and 'labels' directories exist.")
        
        # Get all image and label files
        self.image_list = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.label_list = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
        
        # Verify that we have the same number of images and labels
        if len(self.image_list) != len(self.label_list):
            print(f"Warning: Number of images ({len(self.image_list)}) does not match "
                  f"number of labels ({len(self.label_list)}). Only paired files will be used.")
            
            # Keep only files that have both image and label
            image_basenames = set(os.path.basename(p).rsplit('.', 1)[0] for p in self.image_list)
            label_basenames = set(os.path.basename(p).rsplit('.', 1)[0] for p in self.label_list)
            
            common_basenames = image_basenames.intersection(label_basenames)
            
            self.image_list = [p for p in self.image_list if os.path.basename(p).rsplit('.', 1)[0] in common_basenames]
            self.label_list = [p for p in self.label_list if os.path.basename(p).rsplit('.', 1)[0] in common_basenames]
    
    def _separate_files(self, number, img_list, txt_list, old_img_list, old_txt_list):
        """
        Separate files into different splits.
        
        Args:
            number: Number of files to separate
            img_list: List to add image files to
            txt_list: List to add label files to
            old_img_list: List of image files to separate from
            old_txt_list: List of label files to separate from
            
        Returns:
            Tuple of (old_img_list, old_txt_list) with separated files removed
        """
        for _ in range(int(number)):
            if not old_txt_list:  # Check if the list is empty
                break
            r = random.randint(0, len(old_txt_list) - 1)
            img_list.append(old_img_list.pop(r))
            txt_list.append(old_txt_list.pop(r))
        return old_img_list, old_txt_list
    
    def _copy_link(self, src, dst):
        """
        Copy or link a file.
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(src):
            os.symlink(os.readlink(src), dst)
        else:
            shutil.copy(src, dst)
    
    def _move_files(self, file_list, dest_dir):
        """
        Move files to the destination directory.
        
        Args:
            file_list: List of files to move
            dest_dir: Destination directory
        """
        os.makedirs(dest_dir, exist_ok=True)
        for item in file_list:
            filename = os.path.basename(item)
            self._copy_link(item, os.path.join(dest_dir, filename))
    
    def _split_and_move_files(self, files_list, images_list, split_type):
        """
        Move files to the appropriate directories.
        
        Args:
            files_list: List of label files
            images_list: List of image files
            split_type: Type of split (train, valid, test)
        """
        # Make sure we have the same number of files and images
        assert len(files_list) == len(images_list), "Number of files and images must match"
        
        # Move files directly
        labels_dir = os.path.join(self.save_dir, split_type, 'labels')
        images_dir = os.path.join(self.save_dir, split_type, 'images')
        
        # Move files
        self._move_files(files_list, labels_dir)
        self._move_files(images_list, images_dir)
    
    def split_dataset(self):
        """Split the dataset into train, validation, and test sets."""
        valid_images, valid_labels = [], []
        test_images, test_labels = [], []
        
        # Create copies of the lists to avoid modifying the originals
        remaining_images = self.image_list.copy()
        remaining_labels = self.label_list.copy()
        
        # Calculate the number of files for each split
        num_valid = int(self.total_files * self.valid_ratio)
        num_test = int(self.total_files * self.test_ratio)
        
        # Separate files for validation and test sets
        remaining_images, remaining_labels = self._separate_files(
            num_valid, valid_images, valid_labels, remaining_images, remaining_labels
        )
        
        remaining_images, remaining_labels = self._separate_files(
            num_test, test_images, test_labels, remaining_images, remaining_labels
        )
        
        # The remaining files go to the training set
        train_images, train_labels = remaining_images, remaining_labels
        
        # Move files to their respective directories
        self._split_and_move_files(train_labels, train_images, 'train')
        self._split_and_move_files(valid_labels, valid_images, 'valid')
        self._split_and_move_files(test_labels, test_images, 'test')
        
        # Print statistics
        print(f"\nDataset split complete:")
        print(f"  - Total files: {self.total_files}")
        print(f"  - Training: {len(train_labels)} ({len(train_labels)/self.total_files*100:.1f}%)")
        print(f"  - Validation: {len(valid_labels)} ({len(valid_labels)/self.total_files*100:.1f}%)")
        print(f"  - Test: {len(test_labels)} ({len(test_labels)/self.total_files*100:.1f}%)")
        print(f"  - Output directory: {self.save_dir}")
        
        return {
            "train": len(train_labels),
            "valid": len(valid_labels),
            "test": len(test_labels),
            "total": self.total_files
        }

# Example usage
if __name__ == "__main__":
    data_path = "/media/java/RRAP03"
    
    splitter = DatasetSplitter(
        data_path=data_path,
        data_location="balanced_dataset/exported100",
        save_dir="split_dataset/exported100",
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15
    )
    
    splitter.split_dataset()