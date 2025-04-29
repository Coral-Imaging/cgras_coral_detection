#!/usr/env/python3

import os
from pathlib import Path
import re
import yaml

class CCVSTileConfigGenerator:
    def __init__(self, target_dir, species, start_date, end_date, tile_ids, output_dir):
        """
        Initialize the CCVS Tile Config Generator.
        
        Args:
            target_dir (str): Directory containing image files.
            species (str): Species to filter for (e.g., 'Amag').
            start_date (str): Start date for filtering in format 'YYYYMMDD'.
            end_date (str): End date for filtering in format 'YYYYMMDD'.
            tile_ids (list): List of tile IDs to include.
            output_dir (str): Directory to save generated YAML files.
        """
        self.target_dir = target_dir
        self.species = species
        self.start_date = start_date
        self.end_date = end_date
        self.tile_ids = tile_ids
        self.output_dir = output_dir
        self.image_list = []
        self.filtered_image_list = []
        self.valid_tiles = []
        
    def parse_filename(self, filename):
        """
        Parses a filename to extract the tile ID, species, date, image number, tank installation, and week number.

        Args:
            filename (str): The filename to parse.

        Returns:
            dict: A dictionary containing the extracted information.
        """
        pattern = r"CGRAS_(?P<species>[A-Za-z]+)_(?P<tank_installation>MIS\d[a-z]?)_(?P<date>\d{8})_(?P<week>w\d)_T(?P<tile_id>\d{2})_(?P<image_number>\d+)\.jpg"
        match = re.match(pattern, filename)
        
        if match:
            return {
                "tile_id": f"T{match.group('tile_id')}",
                "species": match.group("species"),
                "date": match.group("date"),
                "image_number": int(match.group("image_number")),
                "tank_installation": match.group("tank_installation"),
                "week_number": match.group("week")
            }
        else:
            raise ValueError(f"Filename '{filename}' does not match the expected format.")

    def get_valid_tiles(self, filtered_image_list):
        """
        Returns a list of valid tiles that have exactly 24 images for a given date.

        Args:
            filtered_image_list (list): List of filtered image paths.

        Returns:
            list: A list of tuples where each tuple is (tile_id, date) for valid tiles.
        """
        tile_date_image_count = {}

        # Count images for each tile and date
        for image_name in filtered_image_list:
            try:
                parsed_data = self.parse_filename(image_name.name)
                tile_id = parsed_data["tile_id"]
                capture_date = parsed_data["date"]
                key = (tile_id, capture_date)  # Group by tile_id and date
                if key not in tile_date_image_count:
                    tile_date_image_count[key] = []
                tile_date_image_count[key].append(parsed_data["image_number"])
            except ValueError as e:
                print(f"Skipping file during validation: {e}")

        # Collect valid tiles
        valid_tiles = []
        for key, image_numbers in tile_date_image_count.items():
            if len(image_numbers) == 24:
                valid_tiles.append(key)

        return valid_tiles

    def generate_config_filenames(self, valid_tiles, filtered_image_list, species):
        """
        Generates configuration file names for each valid tile and date.

        Args:
            valid_tiles (list): A list of tuples where each tuple is (tile_id, date).
            filtered_image_list (list): List of filtered image paths.
            species (str): The species name.

        Returns:
            list: A list of configuration file names.
        """
        config_filenames = []

        # Create a mapping of (tile_id, date) to tank_installation
        tile_date_to_tank = {}
        for image_name in filtered_image_list:
            try:
                parsed_data = self.parse_filename(image_name.name)
                tile_id = parsed_data["tile_id"]
                capture_date = parsed_data["date"]
                tank_installation = parsed_data["tank_installation"]
                key = (tile_id, capture_date)
                tile_date_to_tank[key] = tank_installation
            except ValueError as e:
                print(f"Skipping file during tank installation extraction: {e}")

        # Generate config filenames
        for tile_id, date in valid_tiles:
            tank_installation = tile_date_to_tank.get((tile_id, date), "unknown")
            config_name = f"tile_sample_{species}_{tank_installation}_{tile_id}_{date}.yaml"
            config_filenames.append(config_name)

        return config_filenames

    def generate_config_files(self, valid_tiles, filtered_image_list, species, output_dir):
        """
        Generates YAML configuration files for each valid tile and date.

        Args:
            valid_tiles (list): A list of tuples where each tuple is (tile_id, date).
            filtered_image_list (list): List of filtered image paths.
            species (str): The species name.
            output_dir (str): The directory to save the generated YAML files.

        Returns:
            None
        """
        # Create a mapping of (tile_id, date) to image files
        tile_date_to_images = {}
        for image_name in filtered_image_list:
            try:
                parsed_data = self.parse_filename(image_name.name)
                tile_id = parsed_data["tile_id"]
                capture_date = parsed_data["date"]
                key = (tile_id, capture_date)
                if key not in tile_date_to_images:
                    tile_date_to_images[key] = []
                tile_date_to_images[key].append(image_name.name)
            except ValueError as e:
                print(f"Skipping file during image mapping: {e}")

        # Generate YAML files
        for tile_id, date in valid_tiles:
            images = tile_date_to_images.get((tile_id, date), [])
            images.sort()  # Ensure images are sorted

            # Prepare the YAML data as a regular dictionary
            yaml_data = {
                "tile_id": tile_id,
                "species": "Acropora", # "acro",
                "settle_time": "2024-10-30",
                "spawning_time": "2024-10-15",
                "season": "2024Oct",
                "num_tabs": [20, 20],
                "tile_size": [280, 280],
                "frame_size": [294, 294],
                "batch_id": f"CG1-{date}2300",
                "batch_time": f"{date[:4]}-{date[4:6]}-{date[6:]} 23:00:00",
                "importer_id": "YAML",
                "operator": "cbrunner",
                "image_files_parent_folder": f"/home/qcr/cgras_data/Source/2024/MIS5_{tile_id}_{date}",
                "images": [
                    {"x": idx % 4, "y": idx // 4, "file": image}
                    for idx, image in enumerate(images)
                ],
            }

            # Write the YAML file
            output_file = Path(output_dir) / f"tile_sample_{species}_MIS5_{tile_id}_{date}.yaml"
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
                if ((self.start_date <= capture_date <= self.end_date) and 
                    (image_species == self.species) and 
                    (image_tile_id in self.tile_ids)):
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
            print("\nThe following tiles have exactly 24 images for a given date:")
            for tile_id, date in self.valid_tiles:
                print(f"Tile {tile_id} on date {date}")
        else:
            print("No tiles have exactly 24 images for any date.")
        
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
    target_dir = '/home/dtsai/Data/cgras_datasets/cgras_2024_aims_camera_trolley_fixed_filenames/cgras_2024_aims_camera_trolley/corals_spawned_2024_oct'
    species = 'Amag'
    start_date = '20241105'
    end_date = '20241223'
    tile_ids = ['T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T17']
    output_dir = "/home/dtsai/Code/cgras/cgras_settler_counter/scripts_for_ccvs"
    
    # Create and run the generator
    generator = CCVSTileConfigGenerator(
        target_dir=target_dir,
        species=species,
        start_date=start_date,
        end_date=end_date,
        tile_ids=tile_ids,
        output_dir=output_dir
    )
    
    generator.run()


if __name__ == "__main__":
    main()