#! /usr/bin/env python3

""" image_tiler.py
    The ImageTiler class is used to tile images and their corresponding annotations into smaller tiles with overlap.
    The class can be used to tile images and their corresponding annotations into smaller tiles with overlap."""

import os
import re
import glob
import yaml
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from shapely.geometry import Polygon, box, LineString


class ImageTiler:
    def __init__(self, tile_size=(640, 640), overlap_percent=50, data_path="", output_path="", max_files=16382, enforce_containment=False):
        self.tile_width, self.tile_height = tile_size
        self.overlap_percent = overlap_percent / 100
        self.data_path = data_path
        self.output_path = output_path
        self.max_files = max_files
        self.enforce_containment = enforce_containment

        self.classes = self.load_classes()
        self.class_colours = self.default_colours()

        # Ensure output directories exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "labels"), exist_ok=True)

    def load_classes(self):
        yml_path = os.path.join(self.data_path, 'data.yaml')
        with open(yml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']

    def default_colours(self):
        colours = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (144, 65, 2), (0, 128, 0), (192, 192, 192), (0, 100, 0)
        ]
        return {name: colours[i % len(colours)] for i, name in self.classes.items()}

    def is_mostly_contained(self, polygon, x_start, x_end, y_start, y_end):
        polygon_box = box(*polygon.bounds)
        tile_box = box(x_start, y_start, x_end, y_end)
        intersection = polygon.intersection(tile_box)
        return intersection.area > (self.overlap_percent * polygon.area)

    def truncate_polygon(self, polygon, x_start, x_end, y_start, y_end):
        tile_box = box(x_start, y_start, x_end, y_end)
        return polygon.intersection(tile_box)

    def create_polygon_unnormalised(self, parts, img_width, img_height):
        xy_coords = [round(float(p) * img_width) if i % 2 else round(float(p) * img_height) for i, p in enumerate(parts[1:], start=1)]
        polygon_coords = [(xy_coords[i], xy_coords[i + 1]) for i in range(0, len(xy_coords), 2)]
        return Polygon(polygon_coords)

    def normalise_polygon(self, truncated_polygon, class_number, x_start, y_start, width, height):
        if isinstance(truncated_polygon, Polygon):
            x_coords, y_coords = truncated_polygon.exterior.coords.xy
        elif isinstance(truncated_polygon, LineString):
            x_coords, y_coords = truncated_polygon.coords.xy
        else:
            return []
        
        xy = [class_number]
        for c, d in zip(x_coords, y_coords):
            xy.extend([(c - x_start) / width, (d - y_start) / height])
        return xy

    def cut_and_save_img(self, np_img, x_start, x_end, y_start, y_end, img_save_path):
        cut_tile = np_img[y_start:y_end, x_start:x_end, :]
        Image.fromarray(cut_tile).save(img_save_path)

    def cut_annotation(self, x_start, x_end, y_start, y_end, lines, imgw, imgh):
        writelines = []
        for line in lines:
            parts = line.split()
            class_number = int(parts[0])
            polygon = self.create_polygon_unnormalised(parts, imgw, imgh)

            if not polygon.is_valid or polygon.is_empty:
                continue

            if self.enforce_containment and not self.is_mostly_contained(polygon, x_start, x_end, y_start, y_end):
                continue

            truncated_polygon = self.truncate_polygon(polygon, x_start, x_end, y_start, y_end)
            if truncated_polygon.is_empty:
                continue

            xyn = self.normalise_polygon(truncated_polygon, class_number, x_start, y_start, self.tile_width, self.tile_height)
            writelines.append(xyn)
        return writelines

    def tile_images(self):
        image_list_path = os.path.join(self.data_path, 'Train.txt')
        with open(image_list_path, 'r') as file:
            images = file.read().splitlines()

        for img_path in images[:self.max_files]:
            img_name = os.path.basename(img_path).rsplit('.', 1)[0]
            full_img_path = os.path.join(self.data_path, img_path)
            full_label_path = full_img_path.replace('images', 'labels').replace('.jpg', '.txt')

            np_img = np.array(Image.open(full_img_path))
            img_h, img_w = np_img.shape[:2]

            step_x = int(self.tile_width * (1 - self.overlap_percent))
            step_y = int(self.tile_height * (1 - self.overlap_percent))

            for x in range(0, img_w - self.tile_width + 1, step_x):
                for y in range(0, img_h - self.tile_height + 1, step_y):
                    tile_img_name = f'{img_name}_{x}_{y}.jpg'
                    tile_img_path = os.path.join(self.output_path, "images", tile_img_name)
                    tile_label_path = os.path.join(self.output_path, "labels", f'{img_name}_{x}_{y}.txt')

                    self.cut_and_save_img(np_img, x, x + self.tile_width, y, y + self.tile_height, tile_img_path)

                    with open(full_label_path, 'r') as label_file:
                        lines = label_file.readlines()

                    tile_labels = self.cut_annotation(x, x + self.tile_width, y, y + self.tile_height, lines, img_w, img_h)
                    with open(tile_label_path, 'w') as tl_file:
                        for line in tile_labels:
                            tl_file.write(' '.join(map(str, line)) + '\n')

    def viz_overlap_and_labels(self, index):
        """
        Visualizes a 9x9 grid of tiled images around a given index, showcasing overlaps and labels.

        Args:
            index (int): Index of the image in the output folder to center the visualization.
        """
        # Get list of tiled images
        image_files = sorted(glob.glob(os.path.join(self.output_path, "images", "*.jpg")))

        if index < 0 or index >= len(image_files):
            print(f"Index {index} is out of bounds. Please select a valid image index (0 to {len(image_files)-1}).")
            return
        
        # Get the center image name
        center_img_path = image_files[index]
        center_img_name = os.path.basename(center_img_path)

        # Extract base name and coordinates using regex to ensure correctness
        match = re.match(r"(.+)_([0-9]+)_([0-9]+)\.jpg", center_img_name)
        if not match:
            print(f"Filename format is not as expected: {center_img_name}. Ensure correct naming convention.")
            return
        
        base_name, center_x, center_y = match.groups()
        center_x, center_y = int(center_x), int(center_y)

        # Define grid size
        grid_size = 9
        half_grid = grid_size // 2
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        
        for i in range(grid_size):
            for j in range(grid_size):
                x_offset = (i - half_grid) * self.tile_width * (1 - self.overlap_percent)
                y_offset = (j - half_grid) * self.tile_height * (1 - self.overlap_percent)

                # Compute corresponding tile coordinates
                tile_x = center_x + int(x_offset)
                tile_y = center_y + int(y_offset)

                tile_img_name = f"{base_name}_{tile_x}_{tile_y}.jpg"
                tile_label_name = f"{base_name}_{tile_x}_{tile_y}.txt"

                tile_img_path = os.path.join(self.output_path, "images", tile_img_name)
                tile_label_path = os.path.join(self.output_path, "labels", tile_label_name)

                # Load image if available
                if os.path.exists(tile_img_path):
                    img = cv.imread(tile_img_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                else:
                    img = np.ones((self.tile_height, self.tile_width, 3), dtype=np.uint8) * 255  # Blank white tile
                
                # Draw labels if available
                if os.path.exists(tile_label_path):
                    with open(tile_label_path, "r") as file:
                        lines = file.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        class_idx = int(parts[0])
                        points = [float(p) for p in parts[1:]]
                        
                        # Convert normalized points to absolute coordinates
                        abs_points = np.array([
                            (int(points[i] * self.tile_width), int(points[i + 1] * self.tile_height))
                            for i in range(0, len(points), 2)
                        ])
                        
                        if len(abs_points) > 0:
                            cv.polylines(img, [abs_points], isClosed=True, color=self.class_colours[self.classes[class_idx]], thickness=2)
                            cv.putText(img, self.classes[class_idx], (abs_points[0][0], abs_points[0][1]), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[self.classes[class_idx]], 2)

                # Display on the grid
                axes[i, j].imshow(img)
                axes[i, j].axis("off")

        plt.suptitle(f"Visualization of Overlapping Tiles for Image {center_img_name}", fontsize=14)
        plt.show()



if __name__ == '__main__':
    data_path = "/media/agoni/RRAP03/exported_data"
    output_path = "/media/agoni/RRAP03/tiled_images_output"

    tiler = ImageTiler(
        tile_size=(640, 640),
        overlap_percent=50,          # 50% overlap
        data_path="/media/agoni/RRAP03/exported_2024_cgras_amag_T01_first10_100quality",
        output_path="/media/agoni/RRAP03/tiled_dataset_containment",
        max_files=16382,
        enforce_containment=False    # Turn off strict annotation containment
    )

    # tiler.tile_images()
    tiler.viz_overlap_and_labels(0) 
