#!/usr/bin/env python3

import os
import random
import re
import argparse
from datetime import datetime

class TileSelector:
    def __init__(self, target_dir, tile_exclude, number_of_tiles):
        """
        Initialize the Tile Selector.
        
        Args:
            target_dir (str): Directory containing tile subfolders
            tile_exclude (list): List of tile IDs to exclude
            number_of_tiles (int): Number of tiles to randomly select
        """
        self.target_dir = target_dir
        self.tile_exclude = tile_exclude
        self.number_of_tiles = number_of_tiles
        
        # For date-based distribution
        self.date_pattern = re.compile(r'_(\d{8})')
    
    def get_all_subfolders(self):
        """
        Get all subfolders in the target directory.
        
        Returns:
            list: List of subfolder names
        """
        try:
            subfolders = [d for d in os.listdir(self.target_dir) 
                         if os.path.isdir(os.path.join(self.target_dir, d))]
            print(f"Found {len(subfolders)} total subfolders in {self.target_dir}")
            return subfolders
        except Exception as e:
            print(f"Error getting subfolders: {str(e)}")
            return []
    
    def filter_subfolders(self, subfolders):
        """
        Filter out subfolders containing excluded tile IDs.
        
        Args:
            subfolders (list): List of subfolder names
            
        Returns:
            list: Filtered list of subfolder names
        """
        filtered = []
        excluded = []
        
        for folder in subfolders:
            should_exclude = any(tile_id in folder for tile_id in self.tile_exclude)
            if should_exclude:
                excluded.append(folder)
            else:
                filtered.append(folder)
        
        print(f"Excluded {len(excluded)} subfolders containing: {', '.join(self.tile_exclude)}")
        print(f"Remaining subfolders: {len(filtered)}")
        
        return filtered
    
    def organize_by_date(self, subfolders):
        """
        Organize subfolders by date to ensure wide distribution.
        
        Args:
            subfolders (list): List of subfolder names
            
        Returns:
            dict: Dictionary mapping dates to lists of subfolders
        """
        date_groups = {}
        
        for folder in subfolders:
            match = self.date_pattern.search(folder)
            if match:
                date = match.group(1)
                if date not in date_groups:
                    date_groups[date] = []
                date_groups[date].append(folder)
            else:
                # For folders without a date, use "unknown" as key
                if "unknown" not in date_groups:
                    date_groups["unknown"] = []
                date_groups["unknown"].append(folder)
        
        print(f"Found subfolders from {len(date_groups)} different dates")
        return date_groups
    
    def select_tiles(self):
        """
        Select tiles with a wide distribution.
        
        Returns:
            list: List of selected subfolder names
        """
        # Get and filter subfolders
        all_subfolders = self.get_all_subfolders()
        filtered_subfolders = self.filter_subfolders(all_subfolders)
        
        # Check if we have enough subfolders
        if len(filtered_subfolders) < self.number_of_tiles:
            print(f"WARNING: Not enough subfolders to select {self.number_of_tiles}. " 
                  f"Will select all {len(filtered_subfolders)} available.")
            return filtered_subfolders
        
        # Organize by date for wide distribution
        date_groups = self.organize_by_date(filtered_subfolders)
        
        # Select tiles with wide distribution across dates
        selected = []
        dates = list(date_groups.keys())
        random.shuffle(dates)  # Randomize date order
        
        # First pass: select one from each date until we reach target number
        for date in dates:
            if len(selected) >= self.number_of_tiles:
                break
                
            folders = date_groups[date]
            selected_folder = random.choice(folders)
            selected.append(selected_folder)
            
            # Remove the selected folder to avoid duplicates
            date_groups[date].remove(selected_folder)
        
        # Second pass: if we still need more, keep selecting from remaining folders
        if len(selected) < self.number_of_tiles:
            remaining = [folder for folders in date_groups.values() for folder in folders]
            random.shuffle(remaining)
            selected.extend(remaining[:self.number_of_tiles - len(selected)])
        
        return selected[:self.number_of_tiles]
    
    def run(self):
        """
        Execute the tile selection process.
        
        Returns:
            list: List of selected subfolder names
        """
        print(f"Starting Tile Selector")
        print(f"Target directory: {self.target_dir}")
        print(f"Excluding tiles: {self.tile_exclude}")
        print(f"Number of tiles to select: {self.number_of_tiles}")
        
        selected_tiles = self.select_tiles()
        
        print(f"\nSelected {len(selected_tiles)} tiles:")
        for tile in selected_tiles:
            print(f"  {tile}")
            
        return selected_tiles


def main():
    parser = argparse.ArgumentParser(description="Select tiles from subfolders with wide distribution")
    parser.add_argument("--target-dir", type=str, 
                        default="/home/dtsai/Data/cgras_datasets/cgras_2024_amag_manual_comparison",
                        help="Directory containing tile subfolders")
    parser.add_argument("--exclude", type=str, nargs="+",
                        default=["T05", "T02"],
                        help="List of tile IDs to exclude")
    parser.add_argument("--number", type=int, default=14,
                        help="Number of tiles to randomly select")
    parser.add_argument("--seed", type=int,
                        help="Random seed for reproducible selection")
    parser.add_argument("--output", type=str,
                        help="Output file to save selected tile list")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    else:
        # Use current time for random seed
        random.seed(datetime.now().timestamp())
    
    selector = TileSelector(args.target_dir, args.exclude, args.number)
    selected_tiles = selector.run()
    
    # Save to output file if specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                for tile in selected_tiles:
                    f.write(f"{tile}\n")
            print(f"Selected tiles saved to {args.output}")
        except Exception as e:
            print(f"Error saving to output file: {str(e)}")
    
    return selected_tiles


if __name__ == "__main__":
    main()

