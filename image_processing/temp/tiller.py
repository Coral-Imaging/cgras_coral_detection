#!/usr/bin/env python3

"""
dataset_tiler.py
Tiles images and their corresponding labels from a split YOLO dataset into smaller overlapping tiles.
Uses the cgras_data.yaml file to locate all train, val, and test images.
"""

import os
import yaml
import glob
import argparse
import multiprocessing
import concurrent.futures
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import (
    Polygon, MultiPolygon, GeometryCollection,
    LineString, Point, box
)


class DatasetTiler:
    """
    A class to tile images and their corresponding labels from a split YOLO dataset.
    Processes all train, val, and test images found in the cgras_data.yaml file.
    Creates a new tiled_dataset directory with the same train/val/test structure.
    """
    
    def __init__(self, yaml_path, output_base_path=None, tile_size=(640, 640), overlap_percent=50, max_workers=None):
        """
        Initialize the DatasetTiler.
        
        Args:
            yaml_path (str): Path to the cgras_data.yaml file.
            output_base_path (str, optional): Base path for the output tiled dataset. If None, uses the parent directory of yaml_path.
            tile_size (tuple): Size of the tiles (width, height).
            overlap_percent (float): Percentage of overlap between adjacent tiles.
            max_workers (int): Maximum number of worker threads to use.
        """
        self.yaml_path = Path(yaml_path)
        self.base_path = self.yaml_path.parent
        if max_workers is None:
            # Use CPU count or a sensible default, but leave some resources for the system
            cpu_count = multiprocessing.cpu_count()
            self.max_workers = max(1, cpu_count - 1)  # Leave at least one CPU for system processes
            print(f"Auto-detected {cpu_count} CPU cores, using {self.max_workers} worker threads")
        else:
            self.max_workers = max_workers
        
        # If output_base_path is not provided, use the parent directory of yaml_path
        if output_base_path is None:
            self.output_base_path = self.base_path
        else:
            self.output_base_path = Path(output_base_path)
            
        self.tiled_dataset_path = self.output_base_path / "tiled_dataset"
        
        self.tile_width, self.tile_height = tile_size
        self.overlap_percent = overlap_percent / 100
        
        # Load the YAML configuration
        with open(self.yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Store class names for later use
        self.classes = self.config.get('names', {})
        
        # Create the output directory structure
        os.makedirs(self.tiled_dataset_path, exist_ok=True)
    
    def truncate_polygon(self, polygon, x_start, x_end, y_start, y_end):
        """
        Intersect the given polygon with the tile bounding box.
        Always returns a Polygon or MultiPolygon if there's any geometry.
        Line or point intersections are auto-converted to a bounding rectangle.
        
        Args:
            polygon (Polygon): The polygon to truncate.
            x_start, x_end, y_start, y_end (int): Tile boundary coordinates.
            
        Returns:
            Polygon, MultiPolygon or None: The truncated polygon or None if no intersection.
        """
        tile_box = box(x_start, y_start, x_end, y_end)
        clipped = polygon.intersection(tile_box)
        
        if clipped.is_empty:
            return None  # No intersection
            
        # If it's already a Polygon, return it
        if isinstance(clipped, Polygon):
            return clipped
            
        # Handle MultiPolygon or GeometryCollection
        if isinstance(clipped, (MultiPolygon, GeometryCollection)):
            polygons = []
            for geom in clipped.geoms:
                if isinstance(geom, Polygon):
                    polygons.append(geom)
                elif isinstance(geom, (LineString, Point)):
                    # Convert line/point to bounding-box polygon
                    env = geom.envelope
                    if isinstance(env, Polygon) and not env.is_empty:
                        polygons.append(env)
                    else:
                        # Buffer to produce a small polygon
                        buffered = geom.buffer(1e-9)
                        if not buffered.is_empty:
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
                return polygons[0]
            else:
                return MultiPolygon(polygons)
                
        # Handle LineString or Point
        if isinstance(clipped, (LineString, Point)):
            env = clipped.envelope
            if isinstance(env, Polygon) and not env.is_empty:
                return env
            else:
                buffered = clipped.buffer(1e-9)
                return None if buffered.is_empty else buffered
                
        return clipped
    
    def create_polygon_unnormalized(self, parts, img_width, img_height):
        """
        Create a polygon from normalized coordinates.
        
        Args:
            parts (list): Normalized polygon coordinates (class_idx, x1, y1, x2, y2, ...).
            img_width (int): Width of the original image.
            img_height (int): Height of the original image.
            
        Returns:
            Polygon: The unnormalized polygon.
        """
        xy_coords = [round(float(p) * img_width) if i % 2 == 0 else round(float(p) * img_height) 
                     for i, p in enumerate(parts[1:])]
        polygon_coords = [(xy_coords[i], xy_coords[i + 1]) for i in range(0, len(xy_coords), 2)]
        return Polygon(polygon_coords)
    
    def normalize_polygon(self, truncated_polygon, class_number, x_start, y_start, width, height):
        """
        Normalize polygon coordinates relative to the tile.
        
        Args:
            truncated_polygon (Polygon or MultiPolygon): The truncated polygon.
            class_number (int): The class number.
            x_start, y_start (int): Tile origin coordinates.
            width, height (int): Tile dimensions.
            
        Returns:
            list: List of normalized polygon coordinates.
        """
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
        """
        Helper function to normalize a single polygon.
        
        Args:
            polygon (Polygon): The polygon to normalize.
            class_number (int): The class number.
            x_start, y_start (int): Tile origin coordinates.
            width, height (int): Tile dimensions.
            
        Returns:
            list: Normalized polygon coordinates.
        """
        x_coords, y_coords = polygon.exterior.coords.xy
        xy = [class_number]
        for c, d in zip(x_coords, y_coords):
            xy.extend([(c - x_start) / width, (d - y_start) / height])
        return xy
    
    def cut_and_save_img(self, img_path, x_start, x_end, y_start, y_end, output_path):
        """
        Cut a tile from an image and save it.
        
        Args:
            img_path (str): Path to the source image.
            x_start, x_end, y_start, y_end (int): The tile boundaries.
            output_path (str): Path to save the tile.
        """
        img = Image.open(img_path)
        np_img = np.array(img)
        
        # Cut the tile
        cut_tile = np_img[y_start:y_end, x_start:x_end]
        
        # Handle case where the tile goes beyond image boundaries
        h, w = cut_tile.shape[:2]
        padded_tile = np.ones((self.tile_height, self.tile_width, 3), dtype=np.uint8) * 255
        padded_tile[:h, :w] = cut_tile
        
        # Save the tile
        Image.fromarray(padded_tile).save(output_path)
    
    def cut_annotation(self, label_path, x_start, x_end, y_start, y_end, img_width, img_height):
        """
        Cut annotations that intersect with the current tile.
        
        Args:
            label_path (str): Path to the label file.
            x_start, x_end, y_start, y_end (int): The tile boundaries.
            img_width, img_height (int): Original image dimensions.
            
        Returns:
            list: List of normalized polygon coordinates for the tile.
        """
        tile_labels = []
        
        # If label file doesn't exist, return empty list
        if not os.path.exists(label_path):
            return tile_labels
            
        # Read the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            class_number = int(parts[0])
            
            # Create polygon from normalized coordinates
            polygon = self.create_polygon_unnormalized(parts, img_width, img_height)
            
            if not polygon.is_valid or polygon.is_empty:
                continue
                
            # Truncate the polygon to the tile boundaries
            truncated_polygon = self.truncate_polygon(polygon, x_start, x_end, y_start, y_end)
            
            if truncated_polygon is None or truncated_polygon.is_empty:
                continue
                
            # Normalize the truncated polygon coordinates
            xyn = self.normalize_polygon(truncated_polygon, class_number, x_start, y_start, 
                                         self.tile_width, self.tile_height)
            
            if xyn and isinstance(xyn[0], list):  # MultiPolygon case
                tile_labels.extend(xyn)  # Add each polygon separately
            elif xyn:  # Single polygon
                tile_labels.append(xyn)
                
        return tile_labels
    
    def _process_single_image(self, img_path, label_dir, split_name, images_dir, labels_dir):
        """
        Process a single image into tiles.
        
        Args:
            img_path (Path): Path to the image file.
            label_dir (Path): Directory containing the label files.
            split_name (str): The split name (train, val, test).
            images_dir (Path): Directory to save tiled images.
            labels_dir (Path): Directory to save tiled labels.
            
        Returns:
            list: List of relative paths to created tile images.
        """
        image_paths = []
        img_path = Path(img_path)
        img_name = img_path.stem
        img_ext = img_path.suffix
        
        # Get corresponding label path
        label_path = Path(label_dir) / f"{img_name}.txt"
        
        # Get image dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Calculate step sizes based on overlap
        step_x = int(self.tile_width * (1 - self.overlap_percent))
        step_y = int(self.tile_height * (1 - self.overlap_percent))
        
        # Tile the image
        for x in range(0, img_width - 1, step_x):
            for y in range(0, img_height - 1, step_y):
                # Adjust x_end and y_end to not exceed image dimensions
                x_end = min(x + self.tile_width, img_width)
                y_end = min(y + self.tile_height, img_height)
                
                # Skip small tiles (less than 50% of tile size)
                if (x_end - x) < self.tile_width // 2 or (y_end - y) < self.tile_height // 2:
                    continue
                    
                # Create tile filenames
                tile_img_name = f"{img_name}_{x}_{y}{img_ext}"
                tile_label_name = f"{img_name}_{x}_{y}.txt"
                
                # Create tile paths
                tile_img_path = images_dir / tile_img_name
                tile_label_path = labels_dir / tile_label_name
                
                # Cut and save the image tile
                self.cut_and_save_img(img_path, x, x_end, y, y_end, tile_img_path)
                
                # Cut and save the label tile
                tile_labels = self.cut_annotation(label_path, x, x_end, y, y_end, img_width, img_height)
                
                if tile_labels:
                    with open(tile_label_path, 'w') as f:
                        for label in tile_labels:
                            f.write(' '.join(map(str, label)) + '\n')
                else:
                    # If no labels, create an empty label file
                    open(tile_label_path, 'w').close()
                    
                # Add to processed images
                rel_path = f"{split_name}/images/{tile_img_name}"
                image_paths.append(rel_path)
        
        return image_paths

    def _process_image_set(self, image_dirs, split_name, max_workers=8):
        """
        Process a set of image directories for a specific split (train/val/test) using thread pool.
        
        Args:
            image_dirs (list): List of image directory paths.
            split_name (str): Name of the split (train, val, or test).
            max_workers (int): Maximum number of worker threads to use.
            
        Returns:
            list: List of relative paths to created tile images.
        """
        # Create directories for the split
        split_dir = self.tiled_dataset_path / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Collect all image files to process
        all_img_files = []
        all_label_dirs = []
        
        # Process each image directory
        for img_dir in image_dirs:
            print(f"Processing {split_name} directory: {img_dir}")
            
            # Convert to absolute path
            abs_img_dir = self.base_path / self.config['path'] / img_dir
            
            # Get all image files
            img_files = sorted(glob.glob(str(abs_img_dir / "*.jpg")))
            img_files.extend(sorted(glob.glob(str(abs_img_dir / "*.png"))))
            
            if not img_files:
                print(f"No images found in {abs_img_dir}")
                continue
            
            # Get corresponding label directory
            label_dir = str(abs_img_dir).replace("/images/", "/labels/")
            
            # Add to the list of files to process
            all_img_files.extend(img_files)
            all_label_dirs.extend([label_dir] * len(img_files))
        
        if not all_img_files:
            print(f"No images found for {split_name}")
            return []
        
        # Process images in parallel
        image_paths = []
        print(f"Processing {len(all_img_files)} images for {split_name} using ThreadPoolExecutor")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a list of futures
            futures = [
                executor.submit(
                    self._process_single_image, 
                    img_path, 
                    label_dir, 
                    split_name, 
                    images_dir, 
                    labels_dir
                )
                for img_path, label_dir in zip(all_img_files, all_label_dirs)
            ]
            
            # Process as they complete with a progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{split_name} Processing"):
                try:
                    result = future.result()
                    image_paths.extend(result)
                except Exception as e:
                    print(f"Error processing image: {e}")
        
        return image_paths
    
    def process(self):
        """
        Process all image directories in the YAML file.
        Creates tiles for train, val, and test sets.
        Updates the YAML file to point to the tiled dataset.
        
        Returns:
            Path: Path to the new YAML file.
        """
        # Initialize containers for the new paths
        train_paths = []
        val_paths = []
        test_paths = []
        
        # Process each split
        if 'train' in self.config:
            print("Processing training images...")
            train_paths = self._process_image_set(self.config['train'], 'train', self.max_workers)
            
        if 'val' in self.config:
            print("Processing validation images...")
            val_paths = self._process_image_set(self.config['val'], 'val', self.max_workers)
            
        if 'test' in self.config:
            print("Processing test images...")
            test_paths = self._process_image_set(self.config['test'], 'test', self.max_workers)
            
        # Create new YAML configuration
        new_config = {
            'names': self.classes,
            'path': 'tiled_dataset',
            'train': ['train/images'] if train_paths else [],
            'val': ['val/images'] if val_paths else [],
            'test': ['test/images'] if test_paths else []
        }
        
        # Save the new YAML configuration
        new_yaml_path = self.output_base_path / "tiled_cgras_data.yaml"
        with open(new_yaml_path, 'w') as f:
            yaml.dump(new_config, f, sort_keys=False)
            
        print(f"Dataset tiling complete. Output saved to {self.tiled_dataset_path}")
        print(f"New YAML configuration saved to {new_yaml_path}")
        print(f"Processed {len(train_paths)} training images, {len(val_paths)} validation images, and {len(test_paths)} test images")
        
        return new_yaml_path


def main():
    parser = argparse.ArgumentParser(description='Tile YOLO datasets based on cgras_data.yaml file.')
    parser.add_argument('--yaml', required=True, help='Path to the cgras_data.yaml file')
    parser.add_argument('--output', default=None, help='Base path for output (default: same as yaml parent directory)')
    parser.add_argument('--tile-size', type=int, nargs=2, default=[640, 640], help='Tile size as width height (default: 640 640)')
    parser.add_argument('--overlap', type=float, default=50, help='Percentage of overlap between tiles (default: 50)')
    parser.add_argument('--max-workers', type=int, default=None, 
                        help='Maximum number of worker threads (default: auto-detect based on CPU count)')
    
    args = parser.parse_args()
    
    # Create and run the tiler
    tiler = DatasetTiler(
        yaml_path=args.yaml,
        output_base_path=args.output,
        tile_size=tuple(args.tile_size),
        overlap_percent=args.overlap,
        max_workers=args.max_workers
    )
    
    tiler.process()


if __name__ == "__main__":
    main()
