#! /usr/bin/env python3

""" random_image_selector.py
    Select random images from a dataset based on spawn months, weeks, species, tile numbers, and position indices.
    The selected images are copied to a specified output directory.
    The script can be run interactively or with the required parameters passed as arguments."""

import random
import shutil
from pathlib import Path

class ImageSelector:
    def __init__(self, base_path, output_path, num_images=100, spawn_months=['nov'], weeks='all', species='all', tile_numbers='all', positions='all'):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.num_images = num_images
        self.spawn_months = spawn_months
        self.weeks = weeks
        self.species = species
        self.tile_numbers = tile_numbers
        self.positions = positions
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_available_weeks(self):
        weeks = set()
        for month in self.spawn_months:
            month_folder = self.base_path / f"corals_spawned_2024_{month}"
            weeks.update(d.name for d in month_folder.iterdir() if d.is_dir())
        return sorted(weeks)

    def select_images(self, interactive=True):
        if interactive:
            months_input = input("Enter spawn months separated by commas (e.g., 'oct,nov') [nov]: ") or "nov"
            self.spawn_months = [m.strip() for m in months_input.split(",")]

            available_weeks = self.get_available_weeks()
            print(f"Available weeks: {', '.join(available_weeks)}")

            weeks_input = input("Enter weeks separated by commas (or 'all') [all]: ") or "all"
            self.weeks = available_weeks if weeks_input == "all" else [w.strip() for w in weeks_input.split(",")]

            species_input = input("Enter species names separated by commas (or 'all') [all]: ") or "all"
            self.species = "all" if species_input == "all" else [sp.strip() for sp in species_input.split(",")]

            tile_input = input("Enter tile numbers separated by commas (or 'all') [all]: ") or "all"
            self.tile_numbers = "all" if tile_input == "all" else [t.strip() for t in tile_input.split(",")]

            pos_input = input("Enter position indices separated by commas (or 'all') [all]: ") or "all"
            self.positions = "all" if pos_input == "all" else [p.strip() for p in pos_input.split(",")]

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

        if total_images < self.num_images:
            if interactive:
                choice = input(f"Only {total_images} images found, continue with all? (y/n): ")
                if choice.lower() != 'y':
                    return []
            self.num_images = total_images

        selected_images = random.sample(image_files, self.num_images)

        for img in selected_images:
            shutil.copy(img, self.output_path / img.name)

        print(f"Selected {self.num_images} images ({(self.num_images / total_images) * 100:.2f}% of total images). Copied to {self.output_path}")
        return selected_images

# Example usage
if __name__ == "__main__":
    # use inputs if running script directly or can be called from another script with the required parameters with interactive=False
    base_path = input("Enter the base path of the dataset: ")
    output_path = input("Enter the path to copy selected images to: ")
    num_images = int(input("Enter number of images to select [100]: ") or "100")
    selector = ImageSelector(base_path, output_path, num_images)
    selector.select_images(interactive=True)
