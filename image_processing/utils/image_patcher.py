import os
import time
import signal
import argparse
import multiprocessing
import numpy as np
import cv2 as cv
import yaml
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from PIL import Image
from shapely.validation import explain_validity
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
from concurrent.futures import ProcessPoolExecutor


class ImagePatcher:
    def __init__(self, yaml_path, output_path, tile_width=640, tile_height=640, 
                 truncate_percent=0.5, max_files=16382, num_workers=None):
        """Initialize the Image Patcher with the given parameters."""
        # Input/output paths
        self.yaml_path = Path(yaml_path)
        self.output_path = Path(output_path)
        
        # Tiling parameters
        self.TILE_WIDTH = tile_width
        self.TILE_HEIGHT = tile_height
        self.TRUNCATE_PERCENT = truncate_percent
        self.max_files = max_files
        self.new_yaml_path = None
        self.use_containment_check = True

        # Workers for parallel processing
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
        print(f"Using {self.num_workers} worker processes")
        
        # Calculate tile overlap based on truncate percentage
        self.TILE_OVERLAP = round((self.TILE_HEIGHT + self.TILE_WIDTH) / 2 * self.TRUNCATE_PERCENT)
        
        # Directory and file counters by dataset
        self.directory_counts = {}
        self.file_counters = {}
        
        # Flag for stopping processing
        self.stop_processing = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load YAML data
        self._load_yaml()
        
    def _load_yaml(self):
        """Load the YAML configuration file and find all dataset paths"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found at {self.yaml_path}")
            
        with open(self.yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            
        self.base_dir = self.yaml_path.parent
        
        # Find all data paths from the yaml file
        self.dataset_paths = []
        
        # Check different possible structures in the YAML file
        for key in ['data', 'train', 'val', 'test']:
            if key in self.yaml_data:
                data_section = self.yaml_data[key]
                
                # Handle list format
                if isinstance(data_section, list):
                    for path in data_section:
                        self.dataset_paths.append((path, self._extract_dataset_name(path, key)))
                
                # Handle dict format
                elif isinstance(data_section, dict):
                    for dataset_name, dataset_info in data_section.items():
                        if isinstance(dataset_info, dict) and 'images' in dataset_info:
                            path = dataset_info['images']
                            self.dataset_paths.append((path, dataset_name))
        
        if not self.dataset_paths:
            raise ValueError("No dataset paths found in YAML file")
            
        print(f"Found {len(self.dataset_paths)} dataset paths in the YAML file")
        
        # Initialize counters for each dataset
        for _, dataset_name in self.dataset_paths:
            self.directory_counts[dataset_name] = 0
            self.file_counters[dataset_name] = 0
            
        # Get all image paths organized by dataset
        self.dataset_images = {}
        for path, dataset_name in self.dataset_paths:
            images_dir = self.base_dir / path
            if not images_dir.exists():
                print(f"Warning: Images directory not found: {images_dir}")
                continue
                
            image_list = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_list.extend(list(images_dir.glob(f"**/*{ext}")))
                
            if not image_list:
                print(f"Warning: No images found in {images_dir}")
                continue
                
            self.dataset_images[dataset_name] = sorted(image_list)
            print(f"Found {len(image_list)} images in dataset '{dataset_name}'")

    def _extract_dataset_name(self, path, split_type=None):
        """
        Extract dataset name from a path like 'train/dataset1/images'
        Include the split type in the name if provided
        """
        parts = path.split('/')
        dataset_name = None
        for i, part in enumerate(parts):
            if i < len(parts) - 1 and parts[i+1] == 'images':
                dataset_name = part
                break
        
        if dataset_name is None:
            dataset_name = parts[0]  # Fallback
            
        # Add split type to the dataset name to make it unique
        if split_type:
            return f"{split_type}_{dataset_name}"
        return dataset_name
        
    def _signal_handler(self, sig, frame):
        """Handle interrupt signals to cleanly stop processing"""
        print("\nCaught signal, cleaning up...")
        self.stop_processing = True
        print("Will exit after current image completes...")
        
    def make_output_directories(self, dataset_name):
        """Create directory structure for saving tiled images and labels."""
        dir_count = self.directory_counts[dataset_name]
        dataset_dir = f"{dataset_name}_{dir_count}"
        
        output_dataset_dir = self.output_path / dataset_dir
        os.makedirs(output_dataset_dir, exist_ok=True)
        
        images_dir = output_dataset_dir / "images"
        labels_dir = output_dataset_dir / "labels"
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        return images_dir, labels_dir, dataset_dir

    def is_mostly_contained(self, polygon, x_start, x_end, y_start, y_end, threshold):
        """Returns true if a Shaply polygon has more then threshold percent in the area of a specified bounding box."""
        polygon_box = box(*polygon.bounds)
        tile_box = box(x_start, y_start, x_end, y_end)
        if not polygon.is_valid:
            explanation = explain_validity(polygon)
            #print(f"Invalid Polygon: {explanation} at {x_start}_{y_start}")
            return False
        if not polygon_box.intersects(tile_box):
            return False
        intersection = polygon.intersection(tile_box)
        return intersection.area > (threshold * polygon.area)

    def truncate_polygon(self, polygon, x_start, x_end, y_start, y_end):
        """Returns a polygon with points constrained to a specified bounding box."""
        tile_box = box(x_start, y_start, x_end, y_end)
        intersection = polygon.intersection(tile_box)
        return intersection

    def create_polygon_unnormalised(self, parts, img_width, img_height):
        """Creates a Polygon from unnormalized part coordinates, as [class_ix, xn, yn ...]"""
        xy_coords = [round(float(p) * img_width) if i % 2 else round(float(p) * img_height) for i, p in enumerate(parts[1:], start=1)]
        polygon_coords = [(xy_coords[i], xy_coords[i + 1]) for i in range(0, len(xy_coords), 2)]
        polygon = Polygon(polygon_coords)
        return polygon

    def normalise_polygon(self, truncated_polygon, class_number, x_start, x_end, y_start, y_end, width, height):
        """Normalize coordinates of a polygon with respect to a specified bounding box."""
        points = []
        if isinstance(truncated_polygon, Polygon):
            x_coords, y_coords = truncated_polygon.exterior.coords.xy
            xy = [class_number]

            for c, d in zip(x_coords, y_coords):
                x_val = 1.0 if c == x_end else (c - x_start) / width
                y_val = 1.0 if d == y_end else (d - y_start) / height
                xy.extend([x_val, y_val])

            points.append(xy)
            
        elif isinstance(truncated_polygon, (MultiPolygon, GeometryCollection)):
            for p in truncated_polygon.geoms:
                points.append(self.normalise_polygon(p, class_number, x_start, x_end, y_start, y_end, width, height))
        return points

    def cut_n_save_img(self, x_start, x_end, y_start, y_end, np_img, img_save_path):
        """Save a tile section of an image given by a bounding box"""
        cut_tile = np.zeros(shape=(self.TILE_WIDTH, self.TILE_HEIGHT, 3), dtype=np.uint8)
        cut_tile[0:self.TILE_HEIGHT, 0:self.TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
        cut_tile_img = Image.fromarray(cut_tile)
        cut_tile_img.save(img_save_path)

    def cut_annotation(self, x_start, x_end, y_start, y_end, lines, imgw, imgh):
        """From instance lines in label file, find objects in the bounding box and return the renormalised xy points if there are any"""
        writeline = []
        incomplete_lines = set()  
        for j, line in enumerate(lines):
            parts = line.split()
            class_number = int(parts[0])
            polygon = self.create_polygon_unnormalised(parts, imgw, imgh)

            if len(parts) < 1 or polygon.is_empty:
                if j not in incomplete_lines: 
                    print(f"line {j} is incomplete")
                    incomplete_lines.add(j)
                    import code
                    code.interact(local=dict(globals(), **locals()))    
                continue
            if self.is_mostly_contained(polygon, x_start, x_end, y_start, y_end, self.TRUNCATE_PERCENT):
                truncated_polygon = self.truncate_polygon(polygon, x_start, x_end, y_start, y_end)
                xyn = self.normalise_polygon(truncated_polygon, class_number, x_start, x_end, y_start, y_end, self.TILE_WIDTH, self.TILE_HEIGHT)
                writeline.append(xyn)
        return writeline

    def calculate_img_section_no(self, img_name):
        """return the number of tiles made from one image depending on tile width and img size"""
        pil_img = Image.open(img_name, mode='r')
        np_img = np.array(pil_img, dtype=np.uint8)
        img = cv.imread(str(img_name))
        imgw, imgh = img.shape[1], img.shape[0]
        # Count number of sections to make
        x_tiles = (imgw + self.TILE_WIDTH - self.TILE_OVERLAP - 1) // (self.TILE_WIDTH - self.TILE_OVERLAP)
        y_tiles = (imgh + self.TILE_HEIGHT - self.TILE_OVERLAP - 1) // (self.TILE_HEIGHT - self.TILE_OVERLAP)
        return x_tiles * y_tiles

    def process_tile(self, args):
        """Process a single tile from an image"""
        img_name, txt_name, test_name, x_start, x_end, y_start, y_end, imgw, imgh = args
        
        # Read image once outside of the loop to avoid repeated disk I/O
        pil_img = Image.open(img_name, mode='r')
        np_img = np.array(pil_img, dtype=np.uint8)
        
        # Cut the tile from the image
        cut_tile = np.zeros(shape=(self.TILE_HEIGHT, self.TILE_WIDTH, 3), dtype=np.uint8)
        cut_tile[0:self.TILE_HEIGHT, 0:self.TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
        
        # Process annotations
        with open(txt_name, 'r') as file:
            lines = file.readlines()
        try:
            writeline = self.cut_annotation(x_start, x_end, y_start, y_end, lines, imgw, imgh)
        except Exception as e:
            return None, None, str(e)
        
        return (test_name, x_start, y_start, cut_tile, writeline)

    def _find_label_path(self, image_path):
        """
        For a given image path, determine the corresponding label path.
        
        Args:
            image_path (Path): Path to an image
            
        Returns:
            Path: Path to the corresponding label file
        """
        # Convert image path to label path
        parent_dir = image_path.parent
        
        if 'images' in str(parent_dir):
            label_dir = str(parent_dir).replace('images', 'labels')
            label_path = Path(label_dir) / f"{image_path.stem}.txt"
            return label_path
            
        # If not found, try one level up
        if 'images' in str(parent_dir.parent):
            label_dir = str(parent_dir.parent).replace('images', 'labels')
            label_path = Path(label_dir) / f"{image_path.stem}.txt"
            return label_path
            
        # Default to same directory but with .txt extension
        return image_path.with_suffix('.txt')

    def save_checkpoint(self):
        """Save a checkpoint file with progress information for each dataset"""
        checkpoint = {
            'timestamp': time.time(),
            'datasets': {}
        }
        
        for dataset_name, img_list in self.dataset_images.items():
            if dataset_name in self.directory_counts:
                checkpoint['datasets'][dataset_name] = {
                    'directory_count': self.directory_counts[dataset_name],
                    'file_counter': self.file_counters[dataset_name],
                    'processed_images': self.processed_counts.get(dataset_name, 0),
                    'total_images': len(img_list)
                }
        
        try:
            checkpoint_path = self.output_path / "tiling_checkpoint.yaml"
            with open(checkpoint_path, 'w') as f:
                yaml.dump(checkpoint, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self):
        """Load checkpoint data if available"""
        checkpoint_path = self.output_path / "tiling_checkpoint.yaml"
        if not checkpoint_path.exists():
            return {}
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = yaml.safe_load(f)
                
            # Restore directory counts and file counters
            for dataset_name, dataset_info in checkpoint.get('datasets', {}).items():
                if dataset_name in self.directory_counts:
                    self.directory_counts[dataset_name] = dataset_info.get('directory_count', 0)
                    self.file_counters[dataset_name] = dataset_info.get('file_counter', 0)
            
            print(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return {}

    def process_all_datasets(self, resume=False, batch_size=100, memory_limit=0.8):
        """
        Process all datasets from the YAML file with memory optimization
        
        Args:
            resume (bool): Whether to resume from a checkpoint
            batch_size (int): Number of images to process before forcing garbage collection
            memory_limit (float, optional): Memory limit in GB before triggering cleanup
        """
        import gc
        import psutil
        import os
        
        # Initialize processed counts
        self.processed_counts = {}
        
        # Load checkpoint if resuming
        if resume:
            checkpoint = self.load_checkpoint()
            processed_images = {}
            for dataset_name, dataset_info in checkpoint.get('datasets', {}).items():
                processed_images[dataset_name] = dataset_info.get('processed_images', 0)
        else:
            processed_images = {dataset_name: 0 for dataset_name in self.dataset_images.keys()}
        
        # Process each dataset
        start_time = time.time()
        
        # Use tqdm for overall dataset progress
        for dataset_name, img_list in tqdm(self.dataset_images.items(), desc="Processing datasets", unit="dataset"):
            # Skip empty datasets
            if not img_list:
                continue
                
            # Get starting point
            start_idx = processed_images.get(dataset_name, 0)
            
            print(f"\nProcessing dataset '{dataset_name}' ({len(img_list)} images) starting from image {start_idx}")
            
            # Process images with tqdm progress bar
            processed_count = 0
            total_tiles = 0
            batch_count = 0
            
            # Add tqdm progress bar for images within the dataset
            for i, img_path in enumerate(tqdm(img_list[start_idx:], desc=f"Images in {dataset_name}", unit="image")):
                idx = i + start_idx
                
                # Check for interrupt
                if self.stop_processing:
                    print(f"Processing stopped at image {idx} of dataset {dataset_name}")
                    break
                
                # Check memory usage and clean up if needed
                if memory_limit is not None:
                    mem_info = psutil.Process(os.getpid()).memory_info()
                    memory_gb = mem_info.rss / (1024 ** 3)  # Convert bytes to GB
                    if memory_gb > memory_limit:
                        print(f"Memory usage ({memory_gb:.2f} GB) exceeds limit ({memory_limit} GB). Cleaning up...")
                        gc.collect()  # Force garbage collection
                
                # Process the image
                # Replace this line to avoid redundant output
                # print(f"Processing image {idx+1}/{len(img_list)}: {img_path.name}")
                tiles_created, status = self.process_image(img_path, dataset_name)
                
                if tiles_created > 0:
                    processed_count += 1
                    total_tiles += tiles_created
                    # Use tqdm.write instead of print to work with progress bars
                    tqdm.write(f"Image {img_path.name}: {status} - Created {tiles_created} tiles")
                else:
                    tqdm.write(f"Image {img_path.name}: {status}")
                
                # Update checkpoint
                processed_images[dataset_name] = idx + 1
                self.processed_counts = processed_images.copy()
                self.save_checkpoint()
                
                # Periodic cleanup after each batch
                batch_count += 1
                if batch_count >= batch_size:
                    batch_count = 0
                    # Only collect garbage if memory usage is high and memory_limit is specified
                    if memory_limit is not None:
                        mem_info = psutil.Process(os.getpid()).memory_info()
                        memory_gb = mem_info.rss / (1024 ** 3)
                        if memory_gb > memory_limit * 0.8:  # 80% of limit
                            gc.collect()  # Force garbage collection
            
            print(f"Completed dataset '{dataset_name}': processed {processed_count} images, created {total_tiles} tiles")
            
            # Force garbage collection between datasets
            gc.collect()
        
        # Generate final YAML file
        self.generate_output_yaml()
        
        # Print final statistics
        end_time = time.time()
        duration = end_time - start_time
        total_processed = sum(self.processed_counts.values())
        
        print(f"\nTotal processing time: {duration:.2f} seconds")
        print(f"Processed {total_processed} images across {len(self.dataset_images)} datasets")


    def process_image(self, img_path, dataset_name):
        """Process a complete image and its annotations, cutting into tiles (memory-optimized)"""
        # Get label path from image path
        label_path = self._find_label_path(img_path)
        if not label_path.exists():
            return 0, f"No label file found at {label_path}"
            
        try:
            # Read image dimensions only first
            img = cv.imread(str(img_path))
            if img is None:
                return 0, f"Failed to read image {img_path}"

            imgw, imgh = img.shape[1], img.shape[0]
            
            # Check if output directory needs to be created or changed
            files_to_add = self.calculate_img_section_no(img_path)
            
            # ── Directory-splitting: honour max_files, unless it is disabled ──
            if self.max_files and self.max_files > 0 and \
               self.file_counters[dataset_name] + files_to_add >= self.max_files:
                print(f"Directory {dataset_name}_{self.directory_counts[dataset_name]} full, creating new directory")
                self.directory_counts[dataset_name] += 1
                self.file_counters[dataset_name] = 0
            
            # Get output directories
            images_dir, labels_dir, _ = self.make_output_directories(dataset_name)
            
            # Image base name
            img_basename = img_path.stem
            
            # Calculate tile grid parameters
            step_size_x = self.TILE_WIDTH - self.TILE_OVERLAP
            step_size_y = self.TILE_HEIGHT - self.TILE_OVERLAP
            
            x_tiles = (imgw - self.TILE_WIDTH + step_size_x) // step_size_x
            y_tiles = (imgh - self.TILE_HEIGHT + step_size_y) // step_size_y
            
            # Handle edge cases
            if x_tiles * step_size_x + self.TILE_WIDTH < imgw:
                x_tiles += 1
            if y_tiles * step_size_y + self.TILE_HEIGHT < imgh:
                y_tiles += 1
            
            # Read label file only once
            try:
                with open(label_path, 'r') as file:
                    label_lines = file.readlines()
            except Exception as e:
                return 0, f"Failed to read label file {label_path}: {e}"
            
            # Process tiles in smaller batches to manage memory
            tiles_created = 0
            batch_size = 25  # Adjust based on your system's memory
            tile_args = []
            
            # Prepare all tile arguments
            for x in range(x_tiles):
                x_start = x * step_size_x
                x_end = min(x_start + self.TILE_WIDTH, imgw)
                
                # Adjust x_start if we're at the edge
                if x_end == imgw:
                    x_start = max(0, imgw - self.TILE_WIDTH)
                    
                for y in range(y_tiles):
                    y_start = y * step_size_y
                    y_end = min(y_start + self.TILE_HEIGHT, imgh)
                    
                    # Adjust y_start if we're at the edge
                    if y_end == imgh:
                        y_start = max(0, imgh - self.TILE_HEIGHT)
                    
                    tile_args.append((x_start, x_end, y_start, y_end))
            
            # Read image with PIL once
            pil_img = Image.open(img_path)
            np_img = np.array(pil_img)
            
            # Process batches of tiles
            for i in range(0, len(tile_args), batch_size):
                batch_args = tile_args[i:i+batch_size]
                
                results = []
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    # Create a list of futures
                    futures = []
                    for x_start, x_end, y_start, y_end in batch_args:
                        # Cut the tile from the image
                        cut_tile = np.zeros(shape=(self.TILE_HEIGHT, self.TILE_WIDTH, 3), dtype=np.uint8)
                        cut_tile[0:self.TILE_HEIGHT, 0:self.TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
                        
                        # Process annotations
                        try:
                            writeline = self.cut_annotation(x_start, x_end, y_start, y_end, label_lines, imgw, imgh)
                        except Exception as e:
                            print(f"Error processing annotations at {x_start},{y_start}: {e}")
                            continue
                        
                        # Save image
                        img_save_path = images_dir / f"{img_basename}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.jpg"
                        cut_tile_img = Image.fromarray(cut_tile)
                        cut_tile_img.save(img_save_path)
                        
                        # Save annotation
                        txt_save_path = labels_dir / f"{img_basename}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.txt"
                        with open(txt_save_path, 'w') as file:
                            for line in writeline:
                                file.write(" ".join(map(str, line)).replace('[', '').replace(']', '').replace(',', '') + "\n")
                        
                        tiles_created += 1
                
                    # Clear memory after each batch
                    import gc
                    gc.collect()
            
            # Update file counter
            self.file_counters[dataset_name] += tiles_created
            
            # Clean up large objects
            del pil_img, np_img
            import gc
            gc.collect()
            
            return tiles_created, "Success"
                
        except Exception as e:
            return 0, f"Error processing image: {str(e)}"


    def calculate_img_section_no(self, img_path):
        """return the number of tiles made from one image depending on tile width and img size"""
        try:
            # Open the image to get dimensions only
            img = cv.imread(str(img_path))
            if img is None:
                return 0
                
            imgw, imgh = img.shape[1], img.shape[0]
            
            # Count number of sections to make
            x_tiles = (imgw + self.TILE_WIDTH - self.TILE_OVERLAP - 1) // (self.TILE_WIDTH - self.TILE_OVERLAP)
            y_tiles = (imgh + self.TILE_HEIGHT - self.TILE_OVERLAP - 1) // (self.TILE_HEIGHT - self.TILE_OVERLAP)
            return x_tiles * y_tiles
        except Exception as e:
            print(f"Error calculating tile count: {e}")
            return 0
        
    def generate_output_yaml(self):
        """
        Build a YAML that lists *all* images/ directories created, grouped
        by their prefix ('train', 'val', 'test', 'data', …). Nothing is
        re-labelled.  Keys with empty lists are omitted.
        """
        output_yaml = self.yaml_data.copy()
        output_yaml['path'] = str(self.output_path.resolve())

        # Collect paths by prefix
        groups = defaultdict(list)

        for img_dir in self.output_path.rglob("images"):
            rel_path  = img_dir.relative_to(self.output_path).as_posix()
            prefix    = img_dir.parent.name.split('_')[0]   # first token before '_'
            groups[prefix].append(rel_path)

        # Merge groups into YAML (keep original prefixes)
        for prefix, paths in groups.items():
            output_yaml[prefix] = sorted(paths)

        yaml_path = self.output_path / "cgras_data.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(output_yaml, f, sort_keys=False)

        self.new_yaml_path = yaml_path

        print(f"\nGenerated tiled dataset YAML: {yaml_path}")
        for key, paths in groups.items():
            print(f"  {key}: {len(paths)} folder(s)")



def main():
    """Command line entry point"""
    # Add argument parser for command line options
    parser = argparse.ArgumentParser(description='Tile images and annotations for YOLO training')
    parser.add_argument('--yaml_path', type=str, required=True, help='Path to YAML file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for tiled images and labels')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--tile_width', type=int, default=640, help='Tile width (default: 640)')
    parser.add_argument('--tile_height', type=int, default=640, help='Tile height (default: 640)')
    parser.add_argument('--truncate', type=float, default=0.5, help='Minimum percentage of object that must be in tile (default: 0.5)')
    parser.add_argument('--max_files', type=int, default=16382, help='Maximum files per directory (default: 16382)')
    parser.add_argument('--resume', action='store_true', help='Resume from the last checkpoint')
    
    args = parser.parse_args()
    
    # Create ImagePatcher instance
    patcher = ImagePatcher(
        yaml_path=args.yaml_path,
        output_path=args.output_dir,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        truncate_percent=args.truncate,
        max_files=args.max_files,
        num_workers=args.workers
    )
    
    # Process all datasets
    patcher.process_all_datasets(resume=args.resume)
    
    print("Done! Tiling process complete.")


if __name__ == "__main__":
    main()