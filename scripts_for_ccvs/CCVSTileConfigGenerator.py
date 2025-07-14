#!/usr/env/python3

import os
from pathlib import Path
import re
import yaml
from datetime import datetime

class CCVSTileConfigGenerator:
    def __init__(self, target_dir, species=None, match_species=False, week='2', match_week=False, 
                 start_date=None, end_date=None, tile_ids=None, output_dir=None):
        """
        Initialize the CCVS Tile Config Generator.
        
        Args:
            target_dir (str): Directory containing image files.
            species (str, optional): Species to filter for (e.g., 'Amag').
            match_species (bool): Whether to match the species name (default: False).
            week (str, optional): Week number for filtering (default: '2').
            match_week (bool): Whether to match the week number (default: False).
            start_date (str, optional): Start date for filtering in format 'YYYYMMDD' or 'YYMMDD'.
            end_date (str, optional): End date for filtering in format 'YYYYMMDD' or 'YYMMDD'.
            tile_ids (list, optional): List of tile IDs to include.
            output_dir (str, optional): Directory to save generated YAML files.
        """
        self.target_dir = target_dir
        self.species = species
        self.match_species = match_species
        self.week = week
        self.match_week = match_week
        self.start_date = self._standardize_date(start_date) if start_date else None
        self.end_date = self._standardize_date(end_date) if end_date else None
        self.tile_ids = tile_ids or []
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_list = []
        self.filtered_image_list = []
        self.valid_tiles = []
    
    def _standardize_date(self, date_str):
        """
        Standardize date format to YYYYMMDD
        
        Args:
            date_str (str): Date string in YYYYMMDD or YYMMDD format
            
        Returns:
            str: Date string in YYYYMMDD format
        """
        if len(date_str) == 6:
            # YYMMDD format - convert to YYYYMMDD by adding '20' prefix
            return f"20{date_str}"
        return date_str
        
    def parse_filename(self, filename):
        """
        Parses a filename to extract the tile ID, species, date, image number, tank installation, and week number.
        Supports both YYYYMMDD and YYMMDD date formats.

        Args:
            filename (str): The filename to parse.

        Returns:
            dict: A dictionary containing the extracted information.
        """
        # New pattern for HD dataset
        hd_pattern = r"HD_(?P<tank_installation>R\d+)_(?P<date>\d{6})_T(?P<tile_id>\d+)_(?P<image_number>\d+)\.jpg"
        
        # Old pattern for QUT's CGRAS dataset
        cgras_pattern = r"CGRAS_(?P<species>[A-Za-z]+)_(?P<tank_installation>MIS\d[a-z]?)_(?P<date>\d{6,8})_(?P<week>w\d)_T(?P<tile_id>\d{2})_(?P<image_number>\d+)\.jpg"
        
        # Try HD pattern first
        match = re.match(hd_pattern, filename)
        
        if match:
            date = match.group("date")
            # Standardize date to 8-digit format if it's 6 digits
            if len(date) == 6:
                date = f"20{date}"
                
            # HD pattern doesn't include species or week, use defaults
            return {
                "tile_id": f"T{match.group('tile_id').zfill(2)}",
                "species": self.species, 
                "date": date,
                "image_number": int(match.group("image_number")),
                "tank_installation": match.group("tank_installation"),
                "week_number": f"w{self.week}"  # Use default week with 'w' prefix
            }
        
        # Try CGRAS pattern if HD pattern doesn't match
        match = re.match(cgras_pattern, filename)
        
        if match:
            date = match.group("date")
            # Standardize date to 8-digit format if it's 6 digits
            if len(date) == 6:
                date = f"20{date}"
                
            # Extract week from filename or use default based on match_week setting
            file_week = match.group("week")
            week_number = file_week if self.match_week else f"w{self.week}"
            species = match.group("species") if self.match_species else self.species
            return {
                "tile_id": f"T{match.group('tile_id')}",
                "species": species, 
                "date": date,
                "image_number": int(match.group("image_number")),
                "tank_installation": match.group("tank_installation"),
                "week_number": week_number
            }
        
        # If neither pattern matches, raise an error
        raise ValueError(f"Filename '{filename}' does not match any of the expected formats.")

    def get_valid_tiles(self, filtered_image_list):
        """
        Returns a list of valid tiles that have exactly 24 images for a given date and tank.

        Args:
            filtered_image_list (list): List of filtered image paths.

        Returns:
            list: A list of tuples where each tuple is (tile_id, date, tank_id) for valid tiles.
        """
        tile_date_tank_image_count = {}

        # Count images for each tile, date, and tank
        for image_name in filtered_image_list:
            try:
                parsed_data = self.parse_filename(image_name.name)
                tile_id = parsed_data["tile_id"]
                capture_date = parsed_data["date"]
                tank_id = parsed_data["tank_installation"]
                
                # Group by tile_id, date, and tank_id for uniqueness
                key = (tile_id, capture_date, tank_id)
                
                if key not in tile_date_tank_image_count:
                    tile_date_tank_image_count[key] = []
                tile_date_tank_image_count[key].append(parsed_data["image_number"])
            except ValueError as e:
                print(f"Skipping file during validation: {e}")

        # Collect valid tiles
        valid_tiles = []
        for key, image_numbers in tile_date_tank_image_count.items():
            if len(image_numbers) == 24:
                valid_tiles.append(key)

        return valid_tiles

    def generate_config_filenames(self, valid_tiles, filtered_image_list, species):
        """
        Generates configuration file names for each valid tile and date.

        Args:
            valid_tiles (list): A list of tuples where each tuple is (tile_id, date, tank_id).
            filtered_image_list (list): List of filtered image paths.
            species (str): The species name.

        Returns:
            list: A list of configuration file names.
        """
        config_filenames = []
        
        # Generate config filenames
        for tile_id, date, tank_id in valid_tiles:
            config_name = f"tile_sample_{species}_{tank_id}_{tile_id}_{date}.yaml"
            config_filenames.append(config_name)

        return config_filenames

    def generate_config_files(self, valid_tiles, filtered_image_list, species, output_dir):
        """
        Generates YAML configuration files for each valid tile and date.

        Args:
            valid_tiles (list): A list of tuples where each tuple is (tile_id, date, tank_id).
            filtered_image_list (list): List of filtered image paths.
            species (str): The species name.
            output_dir (str): The directory to save the generated YAML files.

        Returns:
            None
        """
        # Create a mapping of (tile_id, date, tank_id) to image files
        tile_date_tank_to_images = {}
        
        for image_name in filtered_image_list:
            try:
                parsed_data = self.parse_filename(image_name.name)
                tile_id = parsed_data["tile_id"]
                capture_date = parsed_data["date"]
                tank_id = parsed_data["tank_installation"]
                key = (tile_id, capture_date, tank_id)
                
                # Store images
                if key not in tile_date_tank_to_images:
                    tile_date_tank_to_images[key] = []
                tile_date_tank_to_images[key].append(image_name.name)
                
            except ValueError as e:
                print(f"Skipping file during image mapping: {e}")

        # Generate YAML files
        for tile_id, date, tank_id in valid_tiles:
            images = tile_date_tank_to_images.get((tile_id, date, tank_id), [])
            images.sort()  # Ensure images are sorted

            # Prepare the YAML data as a regular dictionary
            yaml_data = {
                "tile_id": f"{tank_id}_{tile_id}",
                "species": "amag", # "acro",
                "settle_time": "2024-11-30",
                "spawning_time": "2024-11-15",
                "season": "2024Nov",
                "num_tabs": [20, 20],
                "tile_size": [280, 280],
                "frame_size": [294, 294],
                "batch_id": f"CG1-{tank_id}-{date}-2300",
                "batch_time": f"{date[:4]}-{date[4:6]}-{date[6:]} 23:00:00",
                "importer_id": "YAML",
                "operator": "mnordborg",
                "image_files_parent_folder": f"/home/qcr/cgras_data/Source/2024/{tank_id}_{tile_id}_{date}",
                "images": [
                    {"x": idx % 4, "y": idx // 4, "file": image}
                    for idx, image in enumerate(images)
                ],
            }

            # Write the YAML file with tank ID included in the filename
            output_file = Path(output_dir) / f"tile_sample_{species}_{tank_id}_{tile_id}_{date}.yaml"
            with open(output_file, "w") as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)

            print(f"Generated config file: {output_file}")
    
    def run(self):
        """
        Main method to execute the tile config generation workflow.
        """
        # Get list of images in directory recursively
        self.image_list = sorted(Path.rglob(Path(self.target_dir), '*.jpg'))
        
        # Filter image_list based on criteria
        self.filtered_image_list = []
        for image_name in self.image_list:
            try:
                parsed_data = self.parse_filename(image_name.name)
                capture_date = parsed_data["date"]
                image_species = parsed_data["species"]
                image_tile_id = parsed_data["tile_id"]
                
                # Build filter conditions
                date_condition = True
                if self.start_date and self.end_date:
                    date_condition = (self.start_date <= capture_date <= self.end_date)
                
                species_condition = True
                if self.match_species and self.species:
                    species_condition = (image_species == self.species)
                
                tile_condition = True
                if self.tile_ids:
                    tile_condition = (image_tile_id in self.tile_ids)
                
                # Apply all conditions
                if date_condition and species_condition and tile_condition:
                    self.filtered_image_list.append(image_name)
                    
            except ValueError as e:
                print(f"Skipping file: {e}")
        
        # Print sample of filtered list
        if self.filtered_image_list:
            print(f'\nFirst 10 filtered images:')
            for image_name in self.filtered_image_list[:10]:
                print(image_name)
            
            print(f'\nLast 10 filtered images:')
            for image_name in self.filtered_image_list[-10:]:
                print(image_name)
        
        # Get valid tiles
        self.valid_tiles = self.get_valid_tiles(self.filtered_image_list)
        if self.valid_tiles:
            print("\nThe following tiles have exactly 24 images for a given date and tank:")
            for tile_id, date, tank_id in self.valid_tiles:
                print(f"Tile {tile_id} on date {date} in tank {tank_id}")
        else:
            print("No tiles have exactly 24 images for any date/tank combination.")
        
        # Generate configuration file names
        config_filenames = self.generate_config_filenames(self.valid_tiles, self.filtered_image_list, self.species)
        
        # Print the generated configuration file names
        print("\nGenerated configuration file names:")
        for config_name in config_filenames:
            print(config_name)
        
        # Generate configuration files
        self.generate_config_files(self.valid_tiles, self.filtered_image_list, self.species, self.output_dir)
        
        print('\nTile config generation completed!')


def main():
    # Configuration parameters
    # target_dir = '/home/dtsai/Data/cgras_datasets/cgras_2024_aims_camera_trolley_fixed_filenames/cgras_2024_aims_camera_trolley/corals_spawned_2024_oct'
    # target_dir = '/home/dtsai/Data/cgras_datasets/cgras_amag_2024_highdensityexperiment/transfer_3385482_files_7ec9d82a/Aken_20241202'
    target_dir = '/home/dtsai/Data/cgras_datasets/cgras_amag_2024_highdensityexperiment/Aken_20250128_c'

    species = 'amag'
    week  = '12' # 3 months vs 1 month
    match_week = False
    start_date = '250128'
    end_date = '250129'
    tile_ids = ['T01', 'T02', 'T03', 'T4', 'T04', 'T05', 'T06','T7','T07', 'T08', 'T09', 'T10']
    output_dir = "/home/dtsai/Code/cgras/cgras_settler_counter/scripts_for_ccvs/highdensityexperiment_ccvs_config_files/Aken_20250128_c"
    
    # Create and run the generator
    generator = CCVSTileConfigGenerator(
        target_dir=target_dir,
        species=species,
        match_species=False,
        week=week,
        match_week=match_week,
        start_date=start_date,
        end_date=end_date,
        tile_ids=tile_ids,
        output_dir=output_dir
    )
    
    generator.run()


if __name__ == "__main__":
    main()