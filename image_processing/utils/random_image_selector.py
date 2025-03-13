#! /usr/bin/env python3

""" random_image_selector.py
    Select random images from a dataset based on spawn months, weeks, species, tile numbers, and position indices.
    The selected images are copied to a specified output directory.
    FOLDER STRUCTURE MATTERS.
    The script can be run interactively or with the required parameters passed as arguments."""

import os
import random
import shutil
from pathlib import Path


class ImageSelector:
    def __init__(self, ssd_path, base_path="cgras_2024_aims_camera_trolley", output_path="random_images", num_images='all', spawn_months=['nov'], weeks='all', species='all', tile_numbers='all', positions='all'):
        
        self.ssd_path = Path(ssd_path)
        self.base_path = Path(os.path.join(ssd_path, base_path))
        self.output_path = Path(os.path.join(ssd_path, "outputs", output_path))
        self.num_images = num_images
        self.spawn_months = spawn_months
        self.weeks = weeks
        self.species = species
        self.tile_numbers = tile_numbers
        self.positions = positions
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.spawn_months == 'all':
            self.spawn_months = self.get_available_months()

    def get_available_months(self):
        months = [d.name.split('_')[-1] for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith("corals_spawned_2024_")]
        return sorted(months)

    def get_available_weeks(self):
        weeks = set()
        for month in self.spawn_months:
            month_folder = self.base_path / f"corals_spawned_2024_{month}"
            weeks.update(d.name for d in month_folder.iterdir() if d.is_dir())
        return sorted(weeks)

    def select_images(self):
        image_files = []
        for month in self.spawn_months:
            month_folder = self.base_path / f"corals_spawned_2024_{month}"
            for week_folder in month_folder.iterdir():
                if self.weeks == "all" or week_folder.name in self.weeks:
                    for species_folder in week_folder.iterdir():
                        if species_folder.is_dir() and (self.species == "all" or any(sp in species_folder.name for sp in self.species)):
                            for img in species_folder.glob("*.jpg"):
                                parts = img.stem.split('_')
                                tile_num, pos_idx = parts[-2], parts[-1]
                                if (self.tile_numbers == "all" or tile_num in self.tile_numbers) and (self.positions == "all" or pos_idx in self.positions):
                                    image_files.append(img)

        total_images = len(image_files)
        print(f"Found {total_images} images.")

        if total_images < self.num_images or self.num_images == 'all':
            self.num_images = total_images

        selected_images = random.sample(image_files, self.num_images)

        for img in selected_images:
            shutil.copy(img, self.output_path / img.name)

        print(f"Selected {self.num_images} images ({(self.num_images / total_images) * 100:.2f}% of total images). Copied to {self.output_path}")
        return selected_images

# Example usage
if __name__ == "__main__":

    ssd_path = "/media/java/RRAP03"
    output_path = "random_images"
    num_images = 100 # or 'all'
    spawn_months = ['oct'] # or ['nov', 'oct'] or 'all'
    weeks = 'all'
    species = 'Amag' # 'all'
    tile_numbers = 'all'
    positions = 'all'

    selector = ImageSelector(
        ssd_path, 
        output_path=output_path, 
        num_images=num_images, 
        spawn_months=spawn_months, 
        weeks=weeks, 
        species=species, 
        tile_numbers=tile_numbers, 
        positions=positions
    )
    selector.select_images()
