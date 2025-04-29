#!/usr/bin/env python3

# yaml_dir
# use symbolic links

# now for each yaml file, copy the images from the source to the desination directory, defined in the yaml file

# yaml_dir = '/home/dtsai/Code/cgras/cgras_settler_counter/scripts_for_ccvs'

# read all the yaml files
# for each yaml file, create the directory specified in image_files_parent_folder
# then for each image file in the yaml file, create a symbolic link to the image file in the source directory
# the image files are originally specified in the image_directory: target_dir
# target_dir = '/home/dtsai/Data/cgras_datasets/cgras_2024_aims_camera_trolley_fixed_filenames/cgras_2024_aims_camera_trolley/corals_spawned_2024_oct'



#!/usr/bin/env python3

import os
import yaml
from pathlib import Path
import glob
import shutil

class CCVSImageLinker:
    def __init__(self, yaml_dir, target_dir):
        """
        Initialize the CCVS Image Linker.
        
        Args:
            yaml_dir (str): Directory containing YAML configuration files
            target_dir (str): Source directory containing the original image files
        """
        self.yaml_dir = yaml_dir
        self.target_dir = target_dir
        self.yaml_files = []
    
    def find_yaml_files(self):
        """Find all YAML files in the specified directory."""
        yaml_pattern = os.path.join(self.yaml_dir, "*.yaml")
        self.yaml_files = glob.glob(yaml_pattern)
        print(f"Found {len(self.yaml_files)} YAML files")
        return self.yaml_files
    
    def process_single_yaml(self, yaml_path):
        """
        Process a single YAML file to create symbolic links for its images.
        
        Args:
            yaml_path (str): Path to the YAML file to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the YAML file
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Extract destination directory from YAML
            dest_dir = config.get('image_files_parent_folder')
            if not dest_dir:
                print(f"ERROR: No image_files_parent_folder specified in {yaml_path}")
                return False
            
            # Replace username in destination path (qcr → dtsai)
            if dest_dir.startswith('/home/qcr/'):
                dest_dir = dest_dir.replace('/home/qcr/', '/home/dtsai/')
                print(f"Adjusted destination path: {dest_dir}")
            
            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            print(f"Created directory: {dest_dir}")
            
            # Find all image files in the target directory and its subdirectories
            image_files_map = self.find_image_files()
            
            # Process each image in the YAML file
            images = config.get('images', [])
            success_count = 0
            failed_count = 0
            
            for image_data in images:
                # Get the image filename
                image_filename = image_data.get('file')
                if not image_filename:
                    continue
                
                # Check if we found this image in our map
                if image_filename in image_files_map:
                    source_path = image_files_map[image_filename]
                    dest_path = os.path.join(dest_dir, image_filename)
                    
                    # Remove existing link or file if it exists
                    if os.path.exists(dest_path):
                        if os.path.islink(dest_path):
                            os.unlink(dest_path)
                        else:
                            os.remove(dest_path)
                    
                    # Create symbolic link
                    # os.symlink(source_path, dest_path)
                    # print(f"Created symlink: {dest_path} -> {source_path}")
                    shutil.copy(source_path, dest_path)
                    print(f"Copied file: {dest_path} -> {source_path}")
                    success_count += 1
                else:
                    print(f"WARNING: Could not find source file for: {image_filename}")
                    failed_count += 1
            
            print(f"Processed {len(images)} images: {success_count} successful, {failed_count} failed")
            return failed_count == 0
                
        except Exception as e:
            print(f"ERROR processing {yaml_path}: {str(e)}")
            return False
    
    def find_image_files(self):
        """
        Find all image files recursively in the target directory and create a map of filename to full path.
        
        Returns:
            dict: Map of image filenames to their full paths
        """
        print(f"Scanning for image files in: {self.target_dir}")
        image_files = {}
        
        # Handle wildcards in target_dir path
        if '*' in self.target_dir:
            base_dirs = glob.glob(self.target_dir)
            for base_dir in base_dirs:
                if os.path.isdir(base_dir):
                    self._scan_directory(base_dir, image_files)
        else:
            self._scan_directory(self.target_dir, image_files)
                
        print(f"Found {len(image_files)} unique image files")
        return image_files

    def _scan_directory(self, directory, image_files):
        """
        Recursively scan a directory for image files.
        
        Args:
            directory (str): The directory to scan
            image_files (dict): Map to populate with found image files
        """
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    # Only add if we haven't seen this filename before or update if the path is shorter
                    if file not in image_files or len(os.path.join(root, file)) < len(image_files[file]):
                        image_files[file] = os.path.join(root, file)
    
    def process_all_yamls(self):
        """Process all YAML files to create symbolic links for their images."""
        if not self.yaml_files:
            self.find_yaml_files()
        
        success_count = 0
        for yaml_file in self.yaml_files:
            print(f"\nProcessing: {yaml_file}")
            if self.process_single_yaml(yaml_file):
                success_count += 1
        
        print(f"\nProcessed {len(self.yaml_files)} YAML files with {success_count} successes")
        return success_count
    
    def run(self):
        """Main method to execute the image linking workflow."""
        print(f"Starting CCVS Image Linker")
        print(f"YAML directory: {self.yaml_dir}")
        print(f"Source images directory: {self.target_dir}")
        
        self.find_yaml_files()
        self.process_all_yamls()
        
        print("Image linking completed!")


def main():
    yaml_dir = '/home/dtsai/Code/cgras/cgras_settler_counter/scripts_for_ccvs'
    target_dir = '/home/dtsai/Data/cgras_datasets/cgras_2024_aims_camera_trolley_fixed_filenames/cgras_2024_aims_camera_trolley/corals_spawned_2024_oct/*/*'
    
    # Create and run the image linker
    linker = CCVSImageLinker(yaml_dir, target_dir)
    linker.run()


if __name__ == "__main__":
    main()
    