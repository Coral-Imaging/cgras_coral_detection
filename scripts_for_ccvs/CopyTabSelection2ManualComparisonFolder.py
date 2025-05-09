#!/usr/bin/env python3

import os
import shutil
import glob
import re
import argparse


class TabSelectionCopier:
    def __init__(self, tab_selection_dir, target_dir):
        """
        Initialize the Tab Selection File Copier.
        
        Args:
            tab_selection_dir (str): Directory containing tab selection files (PNG and XLSX)
            target_dir (str): Base directory containing target subfolders
        """
        self.tab_selection_dir = tab_selection_dir
        self.target_dir = target_dir
        
        # File pattern info
        self.pattern = r"tile_sample_(\w+)_(\w+)_(T\d+)_(\d{8})"
        
    def find_tab_selection_files(self):
        """
        Find all tab selection PNG and XLSX files in the source directory.
        
        Returns:
            dict: Dictionary mapping (tile, date) to list of files
        """
        # Find all PNG and XLSX files
        all_files = []
        for ext in ["*.png", "*.xlsx"]:
            file_pattern = os.path.join(self.tab_selection_dir, ext)
            all_files.extend(glob.glob(file_pattern))
        
        # Group files by (tile, date) key
        grouped_files = {}
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            match = re.search(self.pattern, file_name)
            
            if match:
                species = match.group(1)
                tank = match.group(2)
                tile = match.group(3)
                date = match.group(4)
                
                key = (species, tank, tile, date)
                
                if key not in grouped_files:
                    grouped_files[key] = []
                grouped_files[key].append(file_path)
        
        return grouped_files
    
    def get_target_subfolder(self, species, tank, tile, date):
        """
        Determine the target subfolder based on tile and date.
        
        Args:
            species (str): Species name
            tank (str): Tank information
            tile (str): Tile ID
            date (str): Date in YYYYMMDD format
            
        Returns:
            str: Path to the target subfolder
        """
        # Format: T02_CG1-202411052300 (Tile_Batch-YYYYMMDDTime)
        return os.path.join(self.target_dir, f"{tile}_CG1-{date}2300")
    
    def copy_files(self, dry_run=False):
        """
        Copy tab selection files to their corresponding target subfolders.
        
        Args:
            dry_run (bool): If True, only show what would be copied without performing the copy
            
        Returns:
            tuple: (total files found, total files copied, subfolders affected)
        """
        grouped_files = self.find_tab_selection_files()
        files_found = sum(len(files) for files in grouped_files.values())
        files_copied = 0
        subfolders_affected = set()
        
        print(f"Found {files_found} files in {len(grouped_files)} groups")
        
        for (species, tank, tile, date), files in grouped_files.items():
            target_subfolder = self.get_target_subfolder(species, tank, tile, date)
            subfolders_affected.add(target_subfolder)
            
            print(f"\nProcessing {species} {tank} {tile} {date}:")
            print(f"  Target subfolder: {target_subfolder}")
            
            # Check if target subfolder exists
            if not os.path.exists(target_subfolder):
                print(f"  WARNING: Target subfolder does not exist: {target_subfolder}")
                print("  Skipping this group")
                continue
            
            # Copy each file
            for file_path in files:
                file_name = os.path.basename(file_path)
                target_path = os.path.join(target_subfolder, file_name)
                
                print(f"  {'Would copy' if dry_run else 'Copying'}: {file_name}")
                
                if not dry_run:
                    try:
                        shutil.copy2(file_path, target_path)
                        files_copied += 1
                    except Exception as e:
                        print(f"  ERROR: Failed to copy {file_path}: {str(e)}")
                else:
                    # Count as copied for dry run
                    files_copied += 1
        
        return files_found, files_copied, len(subfolders_affected)
    
    def run(self, dry_run=False):
        """
        Execute the tab selection file copying process.
        
        Args:
            dry_run (bool): If True, only show what would be copied without performing the copy
        """
        print(f"{'DRY RUN MODE: ' if dry_run else ''}Starting Tab Selection File Copier")
        print(f"Source directory: {self.tab_selection_dir}")
        print(f"Target base directory: {self.target_dir}")
        
        files_found, files_copied, subfolders = self.copy_files(dry_run)
        
        print(f"\nCopying summary:")
        print(f"Found {files_found} tab selection files")
        print(f"{'Would copy' if dry_run else 'Copied'} {files_copied} files")
        print(f"Affected {subfolders} target subfolders")
        
        if dry_run:
            print("\nThis was a dry run. No files were actually copied.")


def main():
    parser = argparse.ArgumentParser(description="Copy tab selection files to their corresponding target subfolders")
    parser.add_argument("--tab-dir", type=str, 
                        default="/home/dtsai/Code/cgras/cgras_settler_counter/scripts_for_ccvs/output",
                        help="Directory containing tab selection files")
    parser.add_argument("--target-dir", type=str,
                        default="/home/dtsai/Data/cgras_datasets/cgras_2024_amag_manual_comparison",
                        help="Base directory containing target subfolders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without performing the copy")
    
    args = parser.parse_args()
    
    copier = TabSelectionCopier(args.tab_dir, args.target_dir)
    copier.run(args.dry_run)


if __name__ == "__main__":
    main()


