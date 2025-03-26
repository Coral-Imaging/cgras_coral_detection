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
from shapely.geometry import (
    Polygon, MultiPolygon, GeometryCollection,
    LineString, Point, box
)

class ImageTiler:
    def __init__(self, data_path, output_path, tile_size=(640, 640), overlap_percent=50, wanted_classes=None):
        """
        Initialize the ImageTiler class.
        
        Args:
            ssd_path: Base path for the dataset
            data_path: Path to the data directory
            output_path: Path to save the tiled images and labels
            tile_size: Size of the tiles (width, height)
            overlap_percent: Percentage of overlap between tiles
            max_files: Maximum number of files to process
            wanted_classes: Classes to include in the tiled dataset
            use_direct_paths: Whether to use direct paths instead of joining with ssd_path
        """
        self.data_path = data_path
        self.output_path = output_path
        self.tile_width, self.tile_height = tile_size
        self.overlap_percent = overlap_percent / 100
        
        self.classes = self.load_classes()
        self.class_colours = self.default_colours()

        self.wanted_classes = set(wanted_classes) if wanted_classes is not None else set(self.classes.keys())

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "labels"), exist_ok=True)

    def load_classes(self):
        """Load class names from data.yaml file."""
        yml_path = os.path.join(self.data_path, 'data.yaml')
        if not os.path.exists(yml_path):
            print(f"Warning: data.yaml not found at {yml_path}.")
            # Default classes if yaml not found
            return {0: "alive", 1: "dead", 2: "mask_live", 3: "mask_dead"}
            
        with open(yml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']

    def default_colours(self):
        """Generate default colors for visualization."""
        colours = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (144, 65, 2), (0, 128, 0), (192, 192, 192), (0, 100, 0)
        ]
        return {i: colours[i % len(colours)] for i in self.classes.keys()}

    def truncate_polygon(self, polygon, x_start, x_end, y_start, y_end):
        """
        Intersect the given polygon with the tile bounding box, but
        *always* return a Polygon or MultiPolygon if there's any geometry.
        Line or point intersections are auto-converted to a bounding rectangle
        so the downstream code sees them as polygons too.

        Returns None only if there's truly no intersection.
        """
        tile_box = box(x_start, y_start, x_end, y_end)
        clipped = polygon.intersection(tile_box)
        if clipped.is_empty:
            return None  # No intersection at all

        # 1) If it's already a single Polygon, great—just return it
        if isinstance(clipped, Polygon):
            return clipped

        # 2) If it's MultiPolygon or a GeometryCollection, we need to gather sub‐polygons
        if isinstance(clipped, (MultiPolygon, GeometryCollection)):
            polygons = []
            for geom in clipped.geoms:
                if isinstance(geom, Polygon):
                    polygons.append(geom)
                elif isinstance(geom, (LineString, Point)):
                    # Convert line/point to bounding‐box polygon
                    env = geom.envelope  # This yields a Polygon or possibly a LineString if min==max
                    # If envelope is still not a Polygon, buffer slightly or skip
                    if isinstance(env, Polygon) and not env.is_empty:
                        polygons.append(env)
                    else:
                        # Optional: buffer it by a tiny epsilon to produce a small polygon
                        buffered = geom.buffer(1e-9)
                        if not buffered.is_empty:
                            # buffered will often be a polygon
                            polygons.append(buffered)
                elif isinstance(geom, (MultiPolygon, GeometryCollection)):
                    # Recursively handle nested geometry
                    for subgeom in geom.geoms:
                        if isinstance(subgeom, Polygon):
                            polygons.append(subgeom)
                        elif isinstance(subgeom, (LineString, Point)):
                            env = subgeom.envelope
                            if isinstance(env, Polygon) and not env.is_empty:
                                polygons.append(env)
                            else:
                                buffered = subgeom.buffer(1e-9)
                                if not buffered.is_empty:
                                    polygons.append(buffered)

            if not polygons:
                return None
            if len(polygons) == 1:
                return polygons[0]  # Just one sub-polygon
            else:
                return MultiPolygon(polygons)

        # 3) If it's a bare LineString or Point (not inside a collection)
        if isinstance(clipped, (LineString, Point)):
            env = clipped.envelope
            if isinstance(env, Polygon) and not env.is_empty:
                return env
            else:
                # Buffer as fallback
                buffered = clipped.buffer(1e-9)
                return None if buffered.is_empty else buffered

        # 4) Fallback: return whatever clipped is, though it should be
        #    covered by the above if-clauses
        return clipped

    def create_polygon_unnormalised(self, parts, img_width, img_height):
        xy_coords = [round(float(p) * img_width) if i % 2 else round(float(p) * img_height) for i, p in enumerate(parts[1:], start=1)]
        polygon_coords = [(xy_coords[i], xy_coords[i + 1]) for i in range(0, len(xy_coords), 2)]
        return Polygon(polygon_coords)

    def normalise_polygon(self, truncated_polygon, class_number, x_start, y_start, width, height):
        """ Normalize polygon coordinates relative to the tile. """
        
        if truncated_polygon is None or truncated_polygon.is_empty:
            return []

        xy_list = []

        if isinstance(truncated_polygon, MultiPolygon):
            for poly in truncated_polygon.geoms:
                if not poly.is_empty and isinstance(poly, Polygon):
                    xy_list.append(self._normalize_single_polygon(poly, class_number, x_start, y_start, width, height))
            return xy_list

        elif isinstance(truncated_polygon, Polygon):
            return [self._normalize_single_polygon(truncated_polygon, class_number, x_start, y_start, width, height)]

        return []

    def _normalize_single_polygon(self, polygon, class_number, x_start, y_start, width, height):
        """ Helper function to normalize a single polygon. """
        x_coords, y_coords = polygon.exterior.coords.xy
        xy = [class_number]
        for c, d in zip(x_coords, y_coords):
            xy.extend([(c - x_start) / width, (d - y_start) / height])
        return xy

    def cut_and_save_img(self, np_img, x_start, x_end, y_start, y_end, img_save_path):
        cut_tile = np_img[y_start:y_end, x_start:x_end, :]
        
        padded_tile = np.ones((self.tile_height, self.tile_width, 3), dtype=np.uint8) * 255
        h, w = cut_tile.shape[:2]
        padded_tile[:h, :w, :] = cut_tile
        
        Image.fromarray(padded_tile).save(img_save_path)

    def cut_annotation(self, x_start, x_end, y_start, y_end, lines, imgw, imgh):
        writelines = []
        for line in lines:
            parts = line.split()
            class_number = int(parts[0])

            if class_number not in self.wanted_classes:
                continue

            polygon = self.create_polygon_unnormalised(parts, imgw, imgh)

            if not polygon.is_valid or polygon.is_empty:
                continue

            truncated_polygon = self.truncate_polygon(polygon, x_start, x_end, y_start, y_end)

            if truncated_polygon is None or truncated_polygon.is_empty:
                continue

            xyn = self.normalise_polygon(truncated_polygon, class_number, x_start, y_start, self.tile_width, self.tile_height)

            if xyn and isinstance(xyn[0], list):  # MultiPolygon case
                writelines.extend(xyn)  # Add each polygon separately
            elif xyn:  # Ensure xyn is not empty before appending
                writelines.append(xyn)  # Add normally
                
        return writelines

    def tile_images(self):
        """Tile images and corresponding labels."""
        # Look for Train.txt or scan image directory
        image_list_path = os.path.join(self.data_path, 'Train.txt')
        
        if os.path.exists(image_list_path):
            # If Train.txt exists, use it
            with open(image_list_path, 'r') as file:
                images = file.read().splitlines()
            
            # Process each image path from Train.txt
            for img_path in images:
                img_name = os.path.basename(img_path).rsplit('.', 1)[0]
                full_img_path = os.path.join(self.data_path, img_path)
                full_label_path = full_img_path.replace('images', 'labels').replace('.jpg', '.txt')
                
                # Skip if image doesn't exist
                if not os.path.exists(full_img_path):
                    print(f"Warning: Image not found: {full_img_path}")
                    continue
                
                # Skip if label doesn't exist
                if not os.path.exists(full_label_path):
                    print(f"Warning: Label not found: {full_label_path}")
                    continue
                
                self._process_single_image(img_name, full_img_path, full_label_path)
        else:
            # If Train.txt doesn't exist, scan image directory
            print(f"Train.txt not found at {image_list_path}. Scanning image directory...")
            image_dir = os.path.join(self.data_path, 'images')
            
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
            # Process each image in the directory
            image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
            
            for img_path in image_files:
                img_name = os.path.basename(img_path).rsplit('.', 1)[0]
                label_path = os.path.join(self.data_path, 'labels', f"{img_name}.txt")
                
                # Skip if label doesn't exist
                if not os.path.exists(label_path):
                    print(f"Warning: Label not found: {label_path}")
                    continue
                
                self._process_single_image(img_name, img_path, label_path)
    
    def _process_single_image(self, img_name, img_path, label_path):
        """Process a single image and its label file."""
        try:
            # Load image
            np_img = np.array(Image.open(img_path))
            img_h, img_w = np_img.shape[:2]
            
            # Calculate step size for tiling
            step_x = int(self.tile_width * (1 - self.overlap_percent))
            step_y = int(self.tile_height * (1 - self.overlap_percent))
            
            # Read label file
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()
            
            # Generate tiles
            for x in range(0, img_w - self.tile_width + 1, step_x):
                for y in range(0, img_h - self.tile_height + 1, step_y):
                    tile_img_name = f'{img_name}_{x}_{y}.jpg'
                    tile_img_path = os.path.join(self.output_path, "images", tile_img_name)
                    tile_label_path = os.path.join(self.output_path, "labels", f'{img_name}_{x}_{y}.txt')
                    
                    # Cut and save image tile
                    self.cut_and_save_img(np_img, x, x + self.tile_width, y, y + self.tile_height, tile_img_path)
                    
                    # Process annotations for this tile
                    tile_labels = self.cut_annotation(x, x + self.tile_width, y, y + self.tile_height, lines, img_w, img_h)
                    
                    # Save annotations
                    with open(tile_label_path, 'w') as tl_file:
                        for line in tile_labels:
                            tl_file.write(' '.join(map(str, line)) + '\n')
        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")

    def viz_overlap_and_labels(self, index=None):
        """
        Visualizes a 3x3 grid of tiled images around a given index.
        The chosen index is at the center, and the surrounding images are loaded if available.
        """
        image_files = sorted(glob.glob(os.path.join(self.output_path, "images", "*.jpg")))
        
        if index is None:
            if not image_files:
                print("No tiled images found. Please run tile_images() first.")
                return
            index = np.random.randint(0, len(image_files))

        if index < 0 or index >= len(image_files):
            print(f"Index {index} is out of bounds. Please select a valid image index (0 to {len(image_files)-1}).")
            return

        center_img_path = image_files[index]
        center_img_name = os.path.basename(center_img_path)

        match = re.match(r"(.+)_([0-9]+)_([0-9]+)\.jpg", center_img_name)
        if not match:
            print(f"Filename format is not as expected: {center_img_name}. Ensure correct naming convention.")
            return
        
        base_name, center_x, center_y = match.groups()
        center_x, center_y = int(center_x), int(center_y)

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        step = int(self.tile_width * (1 - self.overlap_percent))

        for i in range(3):
            for j in range(3):
                x_offset = (i - 1) * step
                y_offset = (j - 1) * step
                tile_x = center_x + x_offset
                tile_y = center_y + y_offset

                tile_img_name = f"{base_name}_{tile_x}_{tile_y}.jpg"
                tile_label_name = f"{base_name}_{tile_x}_{tile_y}.txt"

                tile_img_path = os.path.join(self.output_path, "images", tile_img_name)
                tile_label_path = os.path.join(self.output_path, "labels", tile_label_name)

                if os.path.exists(tile_img_path):
                    img = cv.imread(tile_img_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                else:
                    img = np.ones((self.tile_height, self.tile_width, 3), dtype=np.uint8) * 255  # Blank white tile
                
                if os.path.exists(tile_label_path):
                    with open(tile_label_path, "r") as file:
                        lines = file.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        class_idx = int(parts[0])
                        points = [float(p) for p in parts[1:]]
                        abs_points = np.array([
                            (int(points[i] * self.tile_width), int(points[i + 1] * self.tile_height))
                            for i in range(0, len(points), 2)
                        ])
                        if len(abs_points) > 0:
                            cv.polylines(img, [abs_points], isClosed=True, color=self.class_colours[class_idx], thickness=2)
                            cv.putText(img, self.classes[class_idx], (abs_points[0][0], abs_points[0][1]), 
                                       cv.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colours[class_idx], 2)

                axes[i, j].imshow(img)
                axes[i, j].axis("off")

        plt.suptitle(f"3x3 Visualization for Image {center_img_name}", fontsize=14)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == '__main__':
    ssd_path = "/media/java/RRAP03"

    tiler = ImageTiler(
        ssd_path=ssd_path,
        data_path="exported_from_cvat/export_cgras_2024_amag_T01_first10_100quality",
        output_path="image_tiler", 
        tile_size=(640, 640),
        overlap_percent=50,
        wanted_classes=None
    )

    tiler.tile_images()
    # tiler.viz_overlap_and_labels(5)