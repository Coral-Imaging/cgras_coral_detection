import os
import shutil
import argparse

#!/usr/bin/env python3

def copy_folders(source_dir, destination_dir, excluded_folders=None):
    """
    Copy all folders and their contents from source_dir to destination_dir,
    excluding any folders specified in excluded_folders.
    """
    if excluded_folders is None:
        excluded_folders = []
    
    # Make sure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Get all items in the source directory
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        
        # Only process directories
        if os.path.isdir(item_path):
            # Skip excluded folders
            if item in excluded_folders:
                print(f"Skipping excluded folder: {item}")
                continue
            
            # Destination path for this folder
            dest_path = os.path.join(destination_dir, item)
            
            # Copy the folder
            print(f"Copying {item_path} to {dest_path}")
            shutil.copytree(item_path, dest_path, dirs_exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy folders from source to destination, excluding specified folders")
    parser.add_argument('source_dir', help='Source directory to copy from')
    parser.add_argument('destination_dir', help='Destination directory to copy to')
    parser.add_argument('--exclude', nargs='*', default=['test_0', 'test_1', 'test_2'], 
                        help='Folders to exclude (default: test_0, test_1, test_2)')
    
    args = parser.parse_args()
    
    copy_folders(args.source_dir, args.destination_dir, args.exclude)
    print("Copy operation completed!")