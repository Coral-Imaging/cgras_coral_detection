import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from tqdm import tqdm
import concurrent.futures
import threading
from PIL import Image
import cv2
import heapq


class ImageFilterer:
    """
    A class to analyze and filter YOLO training data based on label characteristics.
    Primarily focused on filtering out small labels (polygons) that may be too small to be useful.
    Uses pixel areas instead of normalized areas for easier interpretation.
    """
    
    def __init__(self, yaml_path, output_path=None):
        """
        Initialize the ImageFilterer with paths.
        
        Args:
            yaml_path (str): Path to the YOLO cgras_data.yaml file
            output_path (str, optional): Path where filtered data should be saved
        """
        self.yaml_path = Path(yaml_path)
        self.output_path = Path(output_path) if output_path else None
        self.yaml_data = None
        self.all_label_areas = []  # Now stores pixel areas
        self.label_info = []  # Stores (pixel_area, norm_area, label_path, image_path, class_id, coords, img_width, img_height)
        self.label_count_per_dataset = {}
        self.max_workers = min(32, os.cpu_count() + 4)  # Default thread count
        self.new_yaml_path = None
        
        # Load YAML data
        self._load_yaml()
    
    def _load_yaml(self):
        """Load the YAML configuration file"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found at {self.yaml_path}")
            
        with open(self.yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
                
        # Validate YAML has required fields
        if 'names' not in self.yaml_data:
            raise ValueError("Missing 'names' field in YAML file")
                
        # Determine base directory (where the YAML file is located)
        self.base_dir = self.yaml_path.parent
            
        # Find data paths from all potential sources
        self.data_paths = []
        
        # Check all possible dataset path fields: 'data', 'train', 'val', 'test'
        for key in ['data', 'train', 'val', 'test']:
            if key in self.yaml_data:
                # List of paths
                if isinstance(self.yaml_data[key], list):
                    self.data_paths.extend(self.yaml_data[key])
                # Single string path
                elif isinstance(self.yaml_data[key], str):
                    self.data_paths.append(self.yaml_data[key])
                # Dictionary with paths
                elif isinstance(self.yaml_data[key], dict):
                    for dataset_name, dataset_info in self.yaml_data[key].items():
                        if isinstance(dataset_info, dict) and 'images' in dataset_info:
                            self.data_paths.append(dataset_info['images'])
                    
        if not self.data_paths:
            raise ValueError("No dataset paths found in YAML file")
                
        print(f"Found {len(self.data_paths)} dataset paths in the YAML file")
    
    def _load_filtered_data(self):
        """
        Load data from the filtered output if available.
        Returns True if successful, False otherwise.
        """
        if not self.output_path:
            print("Warning: No output path specified for filtered data.")
            return False
            
        filtered_yaml_path = self.output_path / "cgras_data.yaml"
        self.new_yaml_path = filtered_yaml_path
        if not filtered_yaml_path.exists():
            print("Warning: No filtered data found at", filtered_yaml_path)
            return False
            
        try:
            # Save current state
            original_yaml_path = self.yaml_path
            original_yaml_data = self.yaml_data
            original_base_dir = self.base_dir
            original_data_paths = self.data_paths
            
            # Load filtered data
            self.yaml_path = filtered_yaml_path
            self._load_yaml()
            print(f"Loaded filtered data from {filtered_yaml_path}")
            
            # Check if we have any data paths
            if not self.data_paths:
                # Restore original state
                self.yaml_path = original_yaml_path
                self.yaml_data = original_yaml_data
                self.base_dir = original_base_dir
                self.data_paths = original_data_paths
                print("Warning: No data paths found in filtered data.")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error loading filtered data: {str(e)}")
            return False

    def _calculate_polygon_area(self, points):
        """
        Calculate the area of a polygon using the Shoelace formula.
        
        Args:
            points (numpy.ndarray): 2D array of points [[x1, y1], [x2, y2], ...]
            
        Returns:
            float: Area of the polygon
        """
        # Shoelace formula
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def _get_bounding_box(self, points):
        """
        Get the bounding box (min_x, min_y, max_x, max_y) for a set of polygon points.
        
        Args:
            points (numpy.ndarray): 2D array of points [[x1, y1], [x2, y2], ...]
            
        Returns:
            tuple: (min_x, min_y, max_x, max_y) coordinates
        """
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])
        return (min_x, min_y, max_x, max_y)
        
    def _parse_label_file(self, label_path, image_path=None):
        """
        Parse a single YOLO label file and calculate areas for all polygons.
        
        Args:
            label_path (Path): Path to the label file
            image_path (Path, optional): Path to the corresponding image file
            
        Returns:
            list: List of tuples containing label information
        """
        results = []
        
        try:
            # If image path is provided, get image dimensions
            img_width = None
            img_height = None
            
            if image_path and image_path.exists():
                try:
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                except Exception as e:
                    print(f"Error reading image {image_path}: {str(e)}")
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:  # Need at least class_id + 4 points (8 coords)
                        continue
                        
                    class_id = int(parts[0])
                    
                    # Extract coordinate pairs (normalized 0-1)
                    coords = [float(x) for x in parts[1:]]
                    points = np.array([(coords[i], coords[i+1]) for i in range(0, len(coords), 2)])
                    
                    # Calculate normalized area
                    norm_area = self._calculate_polygon_area(points)
                    
                    # Calculate pixel area if image dimensions are available
                    pixel_area = None
                    if img_width is not None and img_height is not None:
                        # Convert normalized points to pixel coordinates
                        pixel_points = np.zeros_like(points)
                        pixel_points[:, 0] = points[:, 0] * img_width
                        pixel_points[:, 1] = points[:, 1] * img_height
                        
                        # Calculate area in pixels
                        pixel_area = self._calculate_polygon_area(pixel_points)
                    else:
                        # Estimate pixel area using a default image size if needed
                        pixel_area = norm_area * 1000000  # Assume 1000x1000 image
                    
                    # Get bounding box (normalized)
                    bbox = self._get_bounding_box(points)
                    
                    # Store results with full info if image_path is provided
                    if image_path:
                        results.append((pixel_area, norm_area, label_path, image_path, class_id, coords, bbox, img_width, img_height))
                    else:
                        results.append((class_id, pixel_area))
                    
            return results
        except Exception as e:
            print(f"Error parsing label file {label_path}: {str(e)}")
            return []
    
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
    
    def _find_image_path(self, label_path):
        """
        For a given label path, determine the corresponding image path.
        
        Args:
            label_path (Path): Path to a label file
            
        Returns:
            Path: Path to the corresponding image file or None if not found
        """
        # Convert label path to image path
        parent_dir = label_path.parent
        
        # Try common image extensions
        if 'labels' in str(parent_dir):
            image_dir = str(parent_dir).replace('labels', 'images')
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = Path(image_dir) / f"{label_path.stem}{ext}"
                if image_path.exists():
                    return image_path
                    
        # Try one level up
        if 'labels' in str(parent_dir.parent):
            image_dir = str(parent_dir.parent).replace('labels', 'images')
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = Path(image_dir) / f"{label_path.stem}{ext}"
                if image_path.exists():
                    return image_path
                    
        return None
    
    def analyze_dataset_areas(self, use_filtered=False):
        """
        Analyze all datasets to calculate label areas in pixels.
        
        Args:
            use_filtered (bool): Whether to use filtered data instead of original
            
        Returns:
            list: All label pixel areas found across all datasets
        """
        self.all_label_areas = []
        self.label_info = []
        self.label_count_per_dataset = {}

        # Load filtered data if requested
        original_state = None
        if use_filtered:
            # Store original state
            original_state = {
                'yaml_path': self.yaml_path,
                'yaml_data': self.yaml_data,
                'base_dir': self.base_dir,
                'data_paths': self.data_paths
            }
            
            # Load filtered data
            if not self._load_filtered_data():
                print("Proceeding with original data instead.")
        
        for data_path in self.data_paths:
            # Resolve the full path
            full_path = self.base_dir / data_path
            
            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(full_path.glob(f"**/*{ext}")))
                
            if not image_files:
                print(f"No image files found in {full_path}")
                continue
                
            if '/' in data_path:
                parts = data_path.split('/')
                # Get the first component that isn't 'images'
                dataset_name = next((part for part in reversed(parts) if part != 'images'), parts[0])
            else:
                dataset_name = data_path  # Get dataset name from path
                
            dataset_areas = []
            
            print(f"Analyzing {len(image_files)} images in {dataset_name}...")
            
            with tqdm(total=len(image_files), desc=f"Processing {dataset_name}", unit="files") as pbar:
                # Use ThreadPoolExecutor for parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    
                    for img_path in image_files:
                        label_path = self._find_label_path(img_path)
                        if label_path.exists():
                            future = executor.submit(self._parse_label_file, label_path, img_path)
                            futures.append(future)
                        pbar.update(1)
                        
                    # Process results
                    for future in futures:
                        label_results = future.result()
                        for result in label_results:
                            # Each result is (pixel_area, norm_area, label_path, image_path, class_id, coords, bbox, width, height)
                            self.label_info.append(result)
                            pixel_area = result[0]  # Pixel area is first element
                            self.all_label_areas.append(pixel_area)
                            dataset_areas.append(pixel_area)
            
            self.label_count_per_dataset[dataset_name] = len(dataset_areas)
            print(f"Found {len(dataset_areas)} labels in {dataset_name}")
            
        if use_filtered and original_state:
            self.yaml_path = original_state['yaml_path']
            self.yaml_data = original_state['yaml_data']
            self.base_dir = original_state['base_dir']
            self.data_paths = original_state['data_paths']
        
        print(f"Total labels analyzed: {len(self.all_label_areas)}")
        return self.all_label_areas
    
    def plot_area_histogram(self, bins=50, show_percentiles=True, log_x=False, max_area=None, 
                        percentile_limit=None, class_ids=None, use_filtered=False):
        """
        Plot a histogram of label areas in pixels with optional filtering by area and class.
        
        Args:
            bins (int): Number of bins for the histogram
            show_percentiles (bool): Whether to show vertical lines for key percentiles
            log_x (bool): Whether to use log scale for x-axis
            max_area (float, optional): Maximum area in pixels² to include in histogram
            percentile_limit (float, optional): Show only data below this percentile (e.g., 0.05 for bottom 5%)
            class_ids (int or list, optional): Filter by specific class ID(s). If None, show all classes.
            use_filtered (bool): Whether to use filtered data instead of original
            
        Returns:
            numpy.ndarray: The filtered areas array that was plotted
        """
        if use_filtered:
            # Clear existing analysis data
            self.label_info = []
            self.all_label_areas = []
            self.analyze_dataset_areas(use_filtered=True)
        elif not self.label_info:
            # If not using filtered data but no analysis exists, run it
            self.analyze_dataset_areas(use_filtered=False)
            
        if not self.label_info:
            print("Unable to analyze label data.")
            return
        
        # Process class_ids parameter
        if class_ids is not None:
            if isinstance(class_ids, int):
                class_ids = [class_ids]  # Convert single int to list
            elif not isinstance(class_ids, (list, tuple)):
                print("class_ids must be an integer or a list of integers. Showing all classes.")
                class_ids = None
        
        # Get class names for display
        class_names = {}
        if isinstance(self.yaml_data['names'], dict):
            class_names = self.yaml_data['names']
        elif isinstance(self.yaml_data['names'], list):
            class_names = {i: name for i, name in enumerate(self.yaml_data['names'])}
        
        # Filter areas by class
        original_count = len(self.label_info)
        
        if class_ids is not None:
            # Extract data for the specified classes
            filtered_info = [info for info in self.label_info if info[4] in class_ids]
            class_desc = ", ".join([f"{class_id} ({class_names.get(class_id, 'Unknown')})" 
                                for class_id in class_ids])
            print(f"Filtering to show only classes: {class_desc} ({len(filtered_info)}/{original_count} labels, {len(filtered_info)/original_count:.1%})")
        else:
            filtered_info = self.label_info
        
        if not filtered_info:
            print("No labels match the specified class filters.")
            return
        
        # Extract pixel areas from filtered data
        # The pixel area is the first element in each label_info tuple
        areas = np.array([info[0] for info in filtered_info])
        
        # Apply area filters if specified
        if max_area is not None:
            before_count = len(areas)
            areas = areas[areas <= max_area]
            print(f"Filtering to show only areas <= {max_area} pixels² ({len(areas)}/{before_count} labels, {len(areas)/before_count:.1%})")
        
        if percentile_limit is not None:
            if not 0 < percentile_limit < 1:
                print("Percentile limit must be between 0 and 1. Using 0.05 (5%) as default.")
                percentile_limit = 0.05
                
            before_count = len(areas)
            threshold = np.percentile(areas, percentile_limit * 100)
            areas = areas[areas <= threshold]
            print(f"Filtering to show only the bottom {percentile_limit:.1%} of areas (<= {threshold:.1f} pixels², {len(areas)}/{before_count} labels)")
        
        if len(areas) == 0:
            print("No labels remain after filtering.")
            return areas
        
        # Plot histograms
        plt.figure(figsize=(12, 6))
        
        # Create histogram with optional log scale on x-axis
        if log_x and np.min(areas) <= 0:
            # Log scale can't handle zero or negative values
            areas = areas[areas > 0]
            print(f"Removed {original_count - len(areas)} zero or negative values for log scale.")
            
        if log_x:
            plt.xscale('log')
            
        n, bins, patches = plt.hist(areas, bins=bins, alpha=0.7)
        
        if show_percentiles:
            # Calculate key percentiles of the filtered data
            percentiles = {
                '1%': np.percentile(areas, 1),
                '5%': np.percentile(areas, 5),
                '10%': np.percentile(areas, 10),
                '25%': np.percentile(areas, 25),
                '50% (median)': np.percentile(areas, 50)
            }
            
            colors = ['red', 'orange', 'green', 'cyan', 'blue']
            
            # Add vertical lines for percentiles
            for (label, value), color in zip(percentiles.items(), colors):
                plt.axvline(x=value, color=color, linestyle='--', linewidth=1.5, label=f"{label}: {value:.1f} px²")
                
            plt.legend()
        
        # Create appropriate title based on filters
        title = 'Distribution of Label Areas (Pixels)'
        if class_ids is not None:
            if len(class_ids) == 1:
                title += f' - Class {class_ids[0]} ({class_names.get(class_ids[0], "Unknown")})'
            else:
                title += f' - Classes {class_ids}'
        if max_area is not None:
            title += f' <= {max_area} px²'
        if percentile_limit is not None:
            title += f' (Bottom {percentile_limit:.1%})'
        if log_x:
            title += ' - Log Scale X-Axis'
            
        plt.title(title)
        plt.xlabel('Area (pixels²)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Add a log scale Y-axis version for better visualization of distribution
        plt.figure(figsize=(12, 6))
        
        if log_x:
            plt.xscale('log')
            
        plt.hist(areas, bins=bins, alpha=0.7, log=True)
        
        if show_percentiles:
            for (label, value), color in zip(percentiles.items(), colors):
                plt.axvline(x=value, color=color, linestyle='--', linewidth=1.5, label=f"{label}: {value:.1f} px²")
                
            plt.legend()
        
        log_title = f'Distribution of Label Areas (Pixels) - Log Scale Y-Axis'
        if class_ids is not None:
            if len(class_ids) == 1:
                log_title += f' - Class {class_ids[0]} ({class_names.get(class_ids[0], "Unknown")})'
            else:
                log_title += f' - Classes {class_ids}'
        if max_area is not None:
            log_title += f' <= {max_area} px²'
        if percentile_limit is not None:
            log_title += f' (Bottom {percentile_limit:.1%})'
        if log_x:
            log_title += ', Log Scale X-Axis'
            
        plt.title(log_title)
        plt.xlabel('Area (pixels²)')
        plt.ylabel('Frequency (log scale)')
        plt.tight_layout()
        
        plt.show()
        
        # Print some statistics
        stats_title = "Label area statistics (in pixels²)"
        if class_ids is not None:
            if len(class_ids) == 1:
                stats_title += f' for Class {class_ids[0]} ({class_names.get(class_ids[0], "Unknown")})'
            else:
                stats_title += f' for Classes {class_ids}'
        
        print(f"{stats_title}:")
        print(f"  - Count: {len(areas)}")
        print(f"  - Min: {areas.min():.1f}")
        print(f"  - Max: {areas.max():.1f}")
        print(f"  - Mean: {areas.mean():.1f}")
        print(f"  - Median: {np.median(areas):.1f}")
        for p, v in percentiles.items():
            print(f"  - {p}: {v:.1f}")
        
        # If multiple classes, print per-class statistics
        if class_ids is not None and len(class_ids) > 1:
            print("\nPer-class statistics:")
            for class_id in class_ids:
                class_areas = np.array([info[0] for info in filtered_info if info[4] == class_id])
                if len(class_areas) > 0:
                    print(f"\nClass {class_id} ({class_names.get(class_id, 'Unknown')}):")
                    print(f"  - Count: {len(class_areas)}")
                    print(f"  - Min: {class_areas.min():.1f}")
                    print(f"  - Max: {class_areas.max():.1f}")
                    print(f"  - Mean: {class_areas.mean():.1f}")
                    print(f"  - Median: {np.median(class_areas):.1f}")
            
        # return areas
            
    def visualize_smallest_labels(self, n=10, margin=0.02, use_filtered=False):
        """
        Visualize the n smallest labels by cropping and displaying them.
        Shows polygon overlays with transparency for better visibility.
        
        Args:
            n (int): Number of smallest labels to display
            margin (float): Extra margin around the bounding box (ratio of image size)
            use_filtered (bool): Whether to use filtered data instead of original
            
        Returns:
            None: Displays the plots
        """
        if use_filtered:
            # Clear existing analysis data
            self.label_info = []
            self.all_label_areas = []
            self.analyze_dataset_areas(use_filtered=True)
        elif not self.label_info:
            # If not using filtered data but no analysis exists, run it
            self.analyze_dataset_areas(use_filtered=False)
            
        if not self.label_info:
            print("Unable to analyze label data.")
            return
            
        # Get the n smallest labels
        smallest_labels = heapq.nsmallest(n, self.label_info, key=lambda x: x[0])  # Sort by pixel area
        
        # Get class names from YAML
        class_names = {}
        if isinstance(self.yaml_data['names'], dict):
            class_names = self.yaml_data['names']
        elif isinstance(self.yaml_data['names'], list):
            class_names = {i: name for i, name in enumerate(self.yaml_data['names'])}
        
        # Create a subplot grid
        fig, axes = plt.subplots(n, 2, figsize=(14, n * 3))
        
        # Make sure axes is 2D even if n=1
        if n == 1:
            axes = np.array([axes])
        
        for i, (pixel_area, norm_area, label_path, image_path, class_id, coords, bbox, img_width, img_height) in enumerate(smallest_labels):
            try:
                # Open the image file
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Error: Could not read image {image_path}")
                    continue
                
                # Convert BGR to RGB format
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get image dimensions
                img_height, img_width = img.shape[:2]
                
                # Get bounding box coordinates (normalized)
                min_x, min_y, max_x, max_y = bbox
                
                # Convert normalized coordinates to pixel coordinates
                min_x_px = int(min_x * img_width)
                min_y_px = int(min_y * img_height)
                max_x_px = int(max_x * img_width)
                max_y_px = int(max_y * img_height)
                
                # Add margin around the bounding box
                margin_x = int(margin * img_width)
                margin_y = int(margin * img_height)
                
                min_x_px = max(0, min_x_px - margin_x)
                min_y_px = max(0, min_y_px - margin_y)
                max_x_px = min(img_width, max_x_px + margin_x)
                max_y_px = min(img_height, max_y_px + margin_y)
                
                # Create a copy of the full image for annotation
                full_img = img.copy()
                
                # Convert polygon coordinates to pixel values
                points = np.array([(coords[j], coords[j+1]) for j in range(0, len(coords), 2)])
                points_px = np.zeros_like(points)
                points_px[:, 0] = points[:, 0] * img_width
                points_px[:, 1] = points[:, 1] * img_height
                points_px = points_px.astype(np.int32)
                
                # Create a mask of the polygon for transparent overlay
                mask = np.zeros_like(full_img)
                cv2.fillPoly(mask, [points_px], (0, 255, 0))  # Green fill
                
                # Apply the mask with transparency
                alpha = 0.4  # 40% opacity
                full_img = cv2.addWeighted(full_img, 1, mask, alpha, 0)
                
                # Draw polygon outline
                cv2.polylines(full_img, [points_px], True, (255, 0, 0), 2)  # Blue outline
                
                # Draw bounding box for visibility
                cv2.rectangle(full_img, (min_x_px, min_y_px), (max_x_px, max_y_px), (0, 0, 255), 2)  # Red rectangle
                
                # Crop the image for the zoomed view
                cropped_img = img[min_y_px:max_y_px, min_x_px:max_x_px].copy()
                
                # Apply the same transparency polygon to the cropped image
                # First need to offset the polygon coordinates for the cropped image
                cropped_points_px = points_px.copy()
                cropped_points_px[:, 0] = cropped_points_px[:, 0] - min_x_px
                cropped_points_px[:, 1] = cropped_points_px[:, 1] - min_y_px
                
                # Create mask for cropped image
                mask_cropped = np.zeros_like(cropped_img)
                cv2.fillPoly(mask_cropped, [cropped_points_px], (0, 255, 0))  # Green fill
                
                # Apply the mask with transparency
                cropped_img = cv2.addWeighted(cropped_img, 1, mask_cropped, alpha, 0)
                
                # Draw polygon outline on cropped image
                cv2.polylines(cropped_img, [cropped_points_px], True, (255, 0, 0), 2)
                
                # Display images
                class_name = class_names.get(class_id, f"Class {class_id}")
                
                # Create titles
                full_img_title = f"Full Image: {image_path.name}"
                crop_title = f"Label {i+1}: Area={pixel_area:.1f} px², Class={class_name}"
                bbox_dim = f"Bbox: {max_x_px-min_x_px}x{max_y_px-min_y_px} pixels"
                rel_size = f"Image dimensions: {img_width}x{img_height} pixels"
                
                # Display the images
                axes[i, 0].imshow(full_img)
                axes[i, 0].set_title(full_img_title, fontsize=10)
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(cropped_img)
                axes[i, 1].set_title(f"{crop_title}\n{bbox_dim}\n{rel_size}", fontsize=10)
                axes[i, 1].axis('off')
                
            except Exception as e:
                print(f"Error processing label {i+1}: {str(e)}")
                if i < n:
                    axes[i, 0].text(0.5, 0.5, f"Error processing image", 
                                ha='center', va='center')
                    axes[i, 0].axis('off')
                    axes[i, 1].text(0.5, 0.5, f"Error: {str(e)}", 
                                ha='center', va='center')
                    axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def filter_small_labels(self, min_pixel_area, copy_images=True, class_ids=None, incremental=False):
        """
        Filter out labels smaller than the specified area threshold in pixels.
        Can filter specific classes only, and can build upon previous filtering.
        
        Args:
            min_pixel_area (float or list): Minimum polygon area(s) in pixels²
                If list, must match length of class_ids list
            copy_images (bool): Whether to copy image files or just labels
            class_ids (int or list, optional): Class ID(s) to filter. If None, filter all classes.
            incremental (bool): If True, continue filtering from previous results
                    
        Returns:
            dict: Statistics on filtering results
        """
        if not self.output_path:
            raise ValueError("Output path not specified. Set it in __init__ or provide to this method.")
        
        # Determine source paths - either original or previously filtered data
        source_yaml_path = self.yaml_path
        source_base_dir = self.base_dir
        source_data_paths = self.data_paths.copy()  # Copy to avoid modifying original
        
        if incremental:
            filtered_yaml_path = self.output_path / "cgras_data.yaml"
            self.new_yaml_path = filtered_yaml_path
            if filtered_yaml_path.exists():
                print(f"Building upon previous filtered results from {filtered_yaml_path}")
                # Load the previous filtered data as our source
                with open(filtered_yaml_path, 'r') as f:
                    filtered_yaml_data = yaml.safe_load(f)
                    
                # Update source to use the filtered data
                source_yaml_path = filtered_yaml_path
                source_base_dir = self.output_path
                
                # Extract data paths from the filtered YAML
                source_data_paths = []
                if 'data' in filtered_yaml_data:
                    if isinstance(filtered_yaml_data['data'], list):
                        source_data_paths = filtered_yaml_data['data']
                    elif isinstance(filtered_yaml_data['data'], dict):
                        for dataset_name, dataset_info in filtered_yaml_data['data'].items():
                            if 'images' in dataset_info:
                                source_data_paths.append(dataset_info['images'])
                
                # If no paths found in filtered data, fall back to original
                if not source_data_paths:
                    print("Warning: No data paths found in filtered YAML. Using original paths.")
                    source_data_paths = self.data_paths.copy()
            else:
                print(f"Warning: No previous filtered results found at {filtered_yaml_path}. Starting fresh.")
        
        # Handle min_pixel_area and class_ids parameters to support multiple thresholds per class
        class_thresholds = {}  # Map of class_id to threshold
        
        if isinstance(min_pixel_area, (list, tuple)) and isinstance(class_ids, (list, tuple)):
            # Ensure both lists have the same length
            if len(min_pixel_area) != len(class_ids):
                raise ValueError("min_pixel_area and class_ids must have the same length when both are lists")
                
            # Create mapping of class_id to threshold
            for i, class_id in enumerate(class_ids):
                class_thresholds[class_id] = min_pixel_area[i]
                
            # For display purposes
            class_ids = list(class_thresholds.keys())
        else:
            # Process class_ids parameter (standard way)
            if class_ids is not None:
                if isinstance(class_ids, int):
                    class_ids = [class_ids]  # Convert single int to list
                elif not isinstance(class_ids, (list, tuple)):
                    print("class_ids must be an integer or a list of integers. Filtering all classes.")
                    class_ids = None
            
            # If class_ids is specified, create a mapping with the same threshold for all
            if class_ids is not None:
                for class_id in class_ids:
                    class_thresholds[class_id] = min_pixel_area
        
        # Load YAML data from the source
        with open(source_yaml_path, 'r') as f:
            source_yaml_data = yaml.safe_load(f)
                    
        # Get class names for display
        class_names = {}
        if isinstance(source_yaml_data['names'], dict):
            class_names = source_yaml_data['names']
        elif isinstance(source_yaml_data['names'], list):
            class_names = {i: name for i, name in enumerate(source_yaml_data['names'])}
            
        # Print filtering info
        if class_ids is not None:
            if len(class_thresholds) > 0:
                print("Filtering with the following thresholds:")
                for class_id, threshold in class_thresholds.items():
                    class_name = class_names.get(str(class_id) if isinstance(class_names, dict) else class_id, f"Class {class_id}")
                    print(f"  - Class {class_id} ({class_name}): {threshold} pixels²")
            else:
                class_desc = ", ".join([f"{class_id} ({class_names.get(str(class_id) if isinstance(class_names, dict) else class_id, 'Unknown')})" 
                                    for class_id in class_ids])
                print(f"Filtering only classes: {class_desc} with threshold {min_pixel_area} pixels²")
        else:
            print(f"Filtering all classes with threshold {min_pixel_area} pixels²")
                
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create a copy of the source YAML for the filtered dataset
        filtered_yaml = source_yaml_data.copy()

        # Preserve all dataset path structures in the filtered YAML
        for key in ['data', 'train', 'val', 'test']:
            if key in filtered_yaml:
                # List of paths
                if isinstance(filtered_yaml[key], list):
                    # Keep the same paths but in the new location
                    filtered_yaml[key] = [p for p in filtered_yaml[key]]
                # Single string path
                elif isinstance(filtered_yaml[key], str):
                    # Keep the same path
                    filtered_yaml[key] = filtered_yaml[key]
                # Dictionary structure
                elif isinstance(filtered_yaml[key], dict):
                    # Keep the same structure but paths will be relative to new location
                    pass

        # Set the new 'path' field to the output path
        filtered_yaml['path'] = str(self.output_path.absolute())

        # Statistics
        stats = {
            'total_images_processed': 0,
            'total_labels_processed': 0,
            'total_labels_kept': 0,
            'total_labels_removed': 0,
            'per_dataset': {},
            'per_class': {}
        }
        
        # Initialize per-class statistics
        for class_id in class_ids if class_ids else range(100):  # Assume max 100 classes if None
            stats['per_class'][class_id] = {
                'processed': 0,
                'kept': 0, 
                'removed': 0
            }
        
        # Create a function to filter a single label file with class-specific thresholds
        def _filter_label_file_with_class(src_label_path, dst_label_path, class_thresholds, default_threshold, class_ids=None):
            if not src_label_path.exists():
                return 0, 0, {}
                
            # Find the image file to get dimensions
            image_path = self._find_image_path(src_label_path)
            if not image_path or not image_path.exists():
                # No image found, can't calculate pixel area
                return 0, 0, {}
                
            # Get image dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                return 0, 0, {}
                
            img_height, img_width = img.shape[:2]
            
            kept_lines = []
            kept_count = 0
            removed_count = 0
            
            # For tracking per-class statistics
            class_stats = {}
            
            with open(src_label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:  # Need at least class_id + 4 points (8 coords)
                        kept_lines.append(line)  # Keep malformed lines
                        continue
                    
                    class_id = int(parts[0])
                    
                    # Initialize class stats if needed
                    if class_id not in class_stats:
                        class_stats[class_id] = {'processed': 0, 'kept': 0, 'removed': 0}
                    
                    # If filtering specific classes and this class is not in the list, keep it
                    if class_ids is not None and class_id not in class_ids:
                        kept_lines.append(line)
                        kept_count += 1
                        continue
                    
                    # Count this label as processed for its class
                    class_stats[class_id]['processed'] += 1
                    
                    coords = [float(x) for x in parts[1:]]
                    points = np.array([(coords[i], coords[i+1]) for i in range(0, len(coords), 2)])
                    
                    # Convert to pixel coordinates for area calculation
                    pixel_points = np.zeros_like(points)
                    pixel_points[:, 0] = points[:, 0] * img_width
                    pixel_points[:, 1] = points[:, 1] * img_height
                    
                    # Calculate area in pixels
                    pixel_area = self._calculate_polygon_area(pixel_points)
                    
                    # Get the threshold for this class
                    threshold = class_thresholds.get(class_id, default_threshold)
                    
                    if pixel_area >= threshold:
                        kept_lines.append(line)
                        kept_count += 1
                        class_stats[class_id]['kept'] += 1
                    else:
                        removed_count += 1
                        class_stats[class_id]['removed'] += 1
            
            # Only write the file if there are any kept lines
            if kept_lines:
                os.makedirs(dst_label_path.parent, exist_ok=True)
                with open(dst_label_path, 'w') as f:
                    f.writelines(kept_lines)
            
            return kept_count, removed_count, class_stats
        
        # Process each dataset
        for data_path in source_data_paths:
            # Resolve the full path
            full_path = source_base_dir / data_path
            
            # Get the relative path components to reconstruct in output
            if '/' in data_path:
                rel_path_parts = data_path.split('/')
                dataset_name = rel_path_parts[-3] if len(rel_path_parts) >= 3 else rel_path_parts[0]
            else:
                dataset_name = data_path
            
            # Initialize stats for this dataset
            stats['per_dataset'][dataset_name] = {
                'images_processed': 0,
                'labels_processed': 0,
                'labels_kept': 0,
                'labels_removed': 0,
                'per_class': {}
            }
            
            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(full_path.glob(f"**/*{ext}")))
                
            if not image_files:
                print(f"No image files found in {full_path}")
                continue
                
            # Get output path for this dataset
            out_dataset_path = self.output_path / data_path
            
            print(f"Filtering {len(image_files)} images in {dataset_name}...")
            
            # Create counters and lock for thread-safe updates
            counters = {
                'images_processed': 0,
                'labels_processed': 0,
                'labels_kept': 0,
                'labels_removed': 0,
                'per_class': {}
            }
            counter_lock = threading.Lock()
            
            # Process files with progress bar
            with tqdm(total=len(image_files), desc=f"Filtering {dataset_name}", unit="files") as pbar:
                # Use ThreadPoolExecutor for parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    
                    for img_path in image_files:
                        # Get source and destination paths
                        src_label_path = self._find_label_path(img_path)
                        
                        # Calculate destination paths
                        rel_path = img_path.relative_to(full_path)
                        dst_img_path = out_dataset_path / rel_path
                        dst_label_path = self._find_label_path(dst_img_path)
                        
                        # Copy image if needed
                        if copy_images:
                            os.makedirs(dst_img_path.parent, exist_ok=True)
                            
                            # Submit image copy task
                            def copy_image(src, dst):
                                shutil.copy2(src, dst)
                                return True
                                
                            image_future = executor.submit(copy_image, img_path, dst_img_path)
                            futures.append(image_future)
                        
                        # Process label file if it exists
                        if src_label_path.exists():
                            # Submit label filtering task with class filtering
                            label_future = executor.submit(
                                _filter_label_file_with_class,
                                src_label_path,
                                dst_label_path,
                                class_thresholds,
                                min_pixel_area,  # Default threshold for classes not in class_thresholds
                                class_ids
                            )
                            
                            # Add callback to update counters when done
                            def update_counters(future):
                                kept, removed, class_stats = future.result()
                                with counter_lock:
                                    counters['labels_processed'] += kept + removed
                                    counters['labels_kept'] += kept
                                    counters['labels_removed'] += removed
                                    
                                    # Update per-class statistics
                                    for class_id, stats_dict in class_stats.items():
                                        if class_id not in counters['per_class']:
                                            counters['per_class'][class_id] = {
                                                'processed': 0, 'kept': 0, 'removed': 0
                                            }
                                        
                                        counters['per_class'][class_id]['processed'] += stats_dict['processed']
                                        counters['per_class'][class_id]['kept'] += stats_dict['kept']
                                        counters['per_class'][class_id]['removed'] += stats_dict['removed']
                            
                            label_future.add_done_callback(update_counters)
                            futures.append(label_future)
                        
                        # Update progress
                        with counter_lock:
                            counters['images_processed'] += 1
                        pbar.update(1)
                    
                    # Wait for all tasks to complete
                    concurrent.futures.wait(futures)
            
            # Update statistics
            stats['per_dataset'][dataset_name]['images_processed'] = counters['images_processed']
            stats['per_dataset'][dataset_name]['labels_processed'] = counters['labels_processed']
            stats['per_dataset'][dataset_name]['labels_kept'] = counters['labels_kept']
            stats['per_dataset'][dataset_name]['labels_removed'] = counters['labels_removed']
            stats['per_dataset'][dataset_name]['per_class'] = counters['per_class']
            
            stats['total_images_processed'] += counters['images_processed']
            stats['total_labels_processed'] += counters['labels_processed']
            stats['total_labels_kept'] += counters['labels_kept']
            stats['total_labels_removed'] += counters['labels_removed']
            
            # Update global per-class statistics
            for class_id, class_stats in counters['per_class'].items():
                if class_id not in stats['per_class']:
                    stats['per_class'][class_id] = {'processed': 0, 'kept': 0, 'removed': 0}
                
                stats['per_class'][class_id]['processed'] += class_stats['processed']
                stats['per_class'][class_id]['kept'] += class_stats['kept']
                stats['per_class'][class_id]['removed'] += class_stats['removed']
            
            # Print dataset summary
            print(f"Dataset {dataset_name} filtering complete:")
            print(f"  - Processed {counters['images_processed']} images")
            print(f"  - Kept {counters['labels_kept']} labels, removed {counters['labels_removed']} labels")
        
        # Write the filtered YAML file
        with open(self.output_path / "cgras_data.yaml", 'w') as f:
            yaml.dump(filtered_yaml, f, default_flow_style=False, sort_keys=False)
            self.new_yaml_path = self.output_path / "cgras_data.yaml"
            
        # Print overall summary
        print(f"\nFiltering complete:")
        print(f"  - Processed {stats['total_images_processed']} images across {len(stats['per_dataset'])} datasets")
        print(f"  - Kept {stats['total_labels_kept']} labels, removed {stats['total_labels_removed']} labels")
        
        # Print per-class summary
        if stats['per_class']:
            print("\nPer-class filtering statistics:")
            for class_id, class_stats in sorted(stats['per_class'].items()):
                if class_stats['processed'] > 0:  # Only show classes that were processed
                    class_name = class_names.get(str(class_id) if isinstance(class_names, dict) else class_id, f"Class {class_id}")
                    threshold = class_thresholds.get(class_id, min_pixel_area if class_ids is None or class_id in class_ids else "N/A")
                    print(f"  - Class {class_id} ({class_name}): threshold {threshold} pixels²")
                    print(f"    - Processed: {class_stats['processed']}")
                    print(f"    - Kept: {class_stats['kept']}")
                    print(f"    - Removed: {class_stats['removed']} ({class_stats['removed']/class_stats['processed']*100:.1f}%)")
        
        print(f"\nFiltered data saved to {self.output_path}")
        
        # return stats
    

    def find_label_by_area(self, target_area, visualize=True, use_filtered=False):
        """
        Find and visualize a label with area closest to the target value.
        
        Args:
            target_area (float): Target area in pixels² to find
            visualize (bool): Whether to visualize the found label
            use_filtered (bool): Whether to use filtered data instead of original
            
        Returns:
            tuple: Information about the found label
        """
        if use_filtered:
            # Clear existing analysis data
            self.label_info = []
            self.all_label_areas = []
            self.analyze_dataset_areas(use_filtered=True)
        elif not self.label_info:
            # If not using filtered data but no analysis exists, run it
            self.analyze_dataset_areas(use_filtered=False)
            
        if not self.label_info:
            print("Unable to analyze label data.")
            return
        
        # Find the label with area closest to the target
        closest_label = min(self.label_info, key=lambda x: abs(x[0] - target_area))
        
        pixel_area, norm_area, label_path, image_path, class_id, coords, bbox, img_width, img_height = closest_label
        
        # Get class names from YAML
        class_names = {}
        if isinstance(self.yaml_data['names'], dict):
            class_names = self.yaml_data['names']
        elif isinstance(self.yaml_data['names'], list):
            class_names = {i: name for i, name in enumerate(self.yaml_data['names'])}
        
        class_name = class_names.get(class_id, f"Class {class_id}")
        
        # Print information about the found label
        print(f"Found label with area {pixel_area:.1f} pixels² (target: {target_area} pixels²)")
        print(f"  - Difference: {abs(pixel_area - target_area):.1f} pixels²")
        print(f"  - Class: {class_name}")
        print(f"  - Image: {image_path}")
        print(f"  - Label: {label_path}")
        
        if not visualize:
            return closest_label
        
        # Visualize the found label
        try:
            # Open the image file
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Error: Could not read image {image_path}")
                return closest_label
            
            # Convert BGR to RGB format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            img_height, img_width = img.shape[:2]
            
            # Get bounding box coordinates (normalized)
            min_x, min_y, max_x, max_y = bbox
            
            # Convert normalized coordinates to pixel coordinates
            min_x_px = int(min_x * img_width)
            min_y_px = int(min_y * img_height)
            max_x_px = int(max_x * img_width)
            max_y_px = int(max_y * img_height)
            
            # Add margin around the bounding box (5% of image size)
            margin = 0.05
            margin_x = int(margin * img_width)
            margin_y = int(margin * img_height)
            
            min_x_px = max(0, min_x_px - margin_x)
            min_y_px = max(0, min_y_px - margin_y)
            max_x_px = min(img_width, max_x_px + margin_x)
            max_y_px = min(img_height, max_y_px + margin_y)
            
            # Create a copy of the full image for annotation
            full_img = img.copy()
            
            # Convert polygon coordinates to pixel values
            points = np.array([(coords[j], coords[j+1]) for j in range(0, len(coords), 2)])
            points_px = np.zeros_like(points)
            points_px[:, 0] = points[:, 0] * img_width
            points_px[:, 1] = points[:, 1] * img_height
            points_px = points_px.astype(np.int32)
            
            # Create a mask of the polygon for transparent overlay
            mask = np.zeros_like(full_img)
            cv2.fillPoly(mask, [points_px], (0, 255, 0))  # Green fill
            
            # Apply the mask with transparency
            alpha = 0.4  # 40% opacity
            full_img = cv2.addWeighted(full_img, 1, mask, alpha, 0)
            
            # Draw polygon outline
            cv2.polylines(full_img, [points_px], True, (255, 0, 0), 2)  # Blue outline
            
            # Draw bounding box for visibility
            cv2.rectangle(full_img, (min_x_px, min_y_px), (max_x_px, max_y_px), (0, 0, 255), 2)  # Red rectangle
            
            # Crop the image for the zoomed view
            cropped_img = img[min_y_px:max_y_px, min_x_px:max_x_px].copy()
            
            # Apply the same transparency polygon to the cropped image
            # First need to offset the polygon coordinates for the cropped image
            cropped_points_px = points_px.copy()
            cropped_points_px[:, 0] = cropped_points_px[:, 0] - min_x_px
            cropped_points_px[:, 1] = cropped_points_px[:, 1] - min_y_px
            
            # Create mask for cropped image
            mask_cropped = np.zeros_like(cropped_img)
            cv2.fillPoly(mask_cropped, [cropped_points_px], (0, 255, 0))  # Green fill
            
            # Apply the mask with transparency
            cropped_img = cv2.addWeighted(cropped_img, 1, mask_cropped, alpha, 0)
            
            # Draw polygon outline on cropped image
            cv2.polylines(cropped_img, [cropped_points_px], True, (255, 0, 0), 2)
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Full image
            full_img_title = f"Full Image: {image_path.name}"
            ax1.imshow(full_img)
            ax1.set_title(full_img_title, fontsize=10)
            ax1.axis('off')
            
            # Cropped view
            crop_title = f"Area: {pixel_area:.1f} px² (target: {target_area}), Class: {class_name}"
            bbox_dim = f"Bbox: {max_x_px-min_x_px}x{max_y_px-min_y_px} pixels"
            rel_size = f"Image dimensions: {img_width}x{img_height} pixels"
            ax2.imshow(cropped_img)
            ax2.set_title(f"{crop_title}\n{bbox_dim}\n{rel_size}", fontsize=10)
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing label: {str(e)}")
        
        # return closest_label

    def find_multiple_labels_by_area(self, target_areas, visualize=True, use_filtered=False):
        """
        Find and visualize multiple labels with areas closest to the target values.
        
        Args:
            target_areas (list): List of target areas in pixels²
            visualize (bool): Whether to visualize the found labels
            use_filtered (bool): Whether to use filtered data instead of original
            
        Returns:
            list: Information about the found labels
        """
        if not target_areas:
            print("No target areas provided.")
            return []
        
        # Sort target areas for better visualization
        target_areas = sorted(target_areas)
        
        results = []
        for area in target_areas:
            print(f"\nFinding label closest to {area} pixels²:")
            label = self.find_label_by_area(area, visualize, use_filtered)
            if label:
                results.append(label)
        
        # return results

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter and analyze YOLO dataset labels by pixel area")
    parser.add_argument("yaml_path", help="Path to the YOLO data YAML file")
    parser.add_argument("--output", help="Path where filtered data should be saved")
    parser.add_argument("--analyze", action="store_true", help="Only analyze label areas without filtering")
    parser.add_argument("--min-area", type=float, default=100, help="Minimum label area threshold in pixels² (default: 100)")
    parser.add_argument("--visualize", type=int, default=10, help="Number of smallest labels to visualize (default: 10)")
    parser.add_argument("--threads", type=int, help="Number of worker threads")
    parser.add_argument("--log-x", action="store_true", help="Use log scale for x-axis in histograms")
    
    args = parser.parse_args()
    
    filterer = ImageFilterer(args.yaml_path, args.output)
    
    if args.threads:
        filterer.max_workers = args.threads
    
    if args.analyze or not args.output:
        filterer.analyze_dataset_areas()
        filterer.plot_area_histogram(log_x=args.log_x)
        
        if args.visualize > 0:
            filterer.visualize_smallest_labels(args.visualize)
    else:
        filterer.analyze_dataset_areas()
        filterer.plot_area_histogram(log_x=args.log_x)
        
        if args.visualize > 0:
            filterer.visualize_smallest_labels(args.visualize)
        
        print(f"\nProceed with filtering using min_area={args.min_area} pixels²? (y/n)")
        user_input = input()
        if user_input.lower() == 'y':
            filterer.filter_small_labels(args.min_area)
        else:
            print("Filtering canceled.")