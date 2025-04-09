import os
import random
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import math

# Default classes - can be overridden via command line
DEFAULT_CLASSES = ["Alive", "Dead"]

def load_class_images(base_dir, class_name=None, limit=100):
    """
    Load images from a directory, optionally from a specific class subdirectory
    
    Args:
        base_dir (str): Base directory containing images or class folders
        class_name (str, optional): Name of the class subfolder. If None, load from base_dir directly
        limit (int): Maximum number of images to load
        
    Returns:
        list: List of loaded images
    """
    # If class_name is provided, look in that subdirectory, otherwise use base_dir directly
    image_dir = os.path.join(base_dir, class_name) if class_name else base_dir
    
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return []
    
    # Look for JPEG images with common extensions
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(Path(image_dir).glob(ext)))
        image_files.extend(list(Path(image_dir).glob(ext.upper())))
    
    # Randomly select up to 'limit' images
    if len(image_files) > limit:
        image_files = random.sample(image_files, limit)
    else:
        folder_name = class_name if class_name else os.path.basename(image_dir)
        print(f"Found {len(image_files)} images in {folder_name}")
    
    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            # Check if this is likely a segmentation mask (grayscale)
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                # Convert single-channel grayscale to 3-channel for display
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # For indexed segmentation masks (single channel with values 0-N)
            elif len(img.shape) == 3 and np.unique(img).size < 10 and np.all(np.unique(img) < 255):
                # This might be an indexed segmentation mask - convert to color for better visualization
                # Using a colormap to convert label indices to colors
                img_colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                unique_values = np.unique(img)
                for i, val in enumerate(unique_values):
                    if val == 0:  # Typically 0 is background
                        color = [0, 0, 0]  # Black for background
                    else:
                        # Generate a distinct color for each class
                        color_id = (i * 50) % 255
                        color = [
                            (color_id * 3) % 255,
                            (color_id * 5) % 255,
                            (color_id * 7) % 255
                        ]
                    img_colored[img[:,:,0] == val] = color
                img = img_colored
                
            images.append(img)
    
    return images

def resize_with_padding(image, target_size):
    """
    Resize image while preserving aspect ratio and pad with black borders
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size (width, height)
        
    Returns:
        numpy.ndarray: Resized image with padding
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor to fit within target size while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize the image while maintaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create a black background of the target size
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate position to center the image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place the resized image on the black background
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded

def create_grid(images, grid_size=10, target_size=(100, 100)):
    """
    Create a grid of images
    
    Args:
        images (list): List of images to arrange in a grid
        grid_size (int): Number of images per row/column
        target_size (tuple): Size to resize each image to
        
    Returns:
        numpy.ndarray: Grid of images
    """
    if not images:
        return None
    
    # Resize all images to the target size while preserving aspect ratio
    resized_images = []
    for img in images:
        resized = resize_with_padding(img, target_size)
        resized_images.append(resized)
    
    # Calculate grid dimensions
    num_images = len(resized_images)
    rows = math.ceil(num_images / grid_size)
    
    # Create an empty grid
    grid_height = rows * target_size[1]
    grid_width = grid_size * target_size[0]
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Fill the grid with images
    for i, img in enumerate(resized_images):
        if i >= rows * grid_size:
            break
        
        row = i // grid_size
        col = i % grid_size
        
        y_start = row * target_size[1]
        y_end = y_start + target_size[1]
        x_start = col * target_size[0]
        x_end = x_start + target_size[0]
        
        grid[y_start:y_end, x_start:x_end] = img
    
    return grid

def visualize_classes(input_dir, output_dir, classes=None, samples_per_class=100, 
                     grid_size=10, image_size=(100, 100), single_dir=False, 
                     is_segmentation=False):
    """
    Create grid visualizations for each class or a single directory
    
    Args:
        input_dir (str): Directory containing class folders or images
        output_dir (str): Directory to save grid images
        classes (list, optional): List of class names (subdirectories). If None and not single_dir, 
                                  uses DEFAULT_CLASSES
        samples_per_class (int): Number of samples to include per class
        grid_size (int): Number of images per row in the grid
        image_size (tuple): Size to resize each image to
        single_dir (bool): If True, treats input_dir as a single directory of images
                          rather than a directory containing class folders
        is_segmentation (bool): If True, adds "Segmentation" to output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle single directory mode
    if single_dir:
        # Load images directly from the input directory
        dir_images = load_class_images(input_dir, class_name=None, limit=samples_per_class)
        
        if dir_images:
            # Create and save grid for all images
            grid = create_grid(dir_images, grid_size, image_size)
            if grid is not None:
                # Use directory name as label
                dir_name = os.path.basename(os.path.normpath(input_dir))
                cv2.putText(grid, dir_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Add segmentation to filename if appropriate
                prefix = "segmentation_" if is_segmentation else ""
                output_path = os.path.join(output_dir, f"{prefix}images_grid.jpg")
                cv2.imwrite(output_path, grid)
                print(f"Saved grid to {output_path}")
        else:
            print(f"No images found in {input_dir}")
        return
    
    # Class-based visualization mode
    if classes is None:
        classes = DEFAULT_CLASSES
    
    # Create a combined visualization with samples from all classes
    all_class_images = []
    
    for class_name in tqdm(classes, desc="Processing classes"):
        # Load images for this class
        class_images = load_class_images(input_dir, class_name, samples_per_class)
        
        if not class_images:
            continue
        
        # Add some samples to the combined visualization
        if class_images:
            samples_for_combined = min(len(class_images), 20)  # Take up to 20 samples per class
            all_class_images.extend(random.sample(class_images, samples_for_combined))
        
        # Create and save grid for this class
        grid = create_grid(class_images, grid_size, image_size)
        if grid is not None:
            # Add class name as text on the image
            cv2.putText(grid, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add segmentation to filename if appropriate
            prefix = "segmentation_" if is_segmentation else ""
            output_path = os.path.join(output_dir, f"{prefix}{class_name}_grid.jpg")
            cv2.imwrite(output_path, grid)
            print(f"Saved grid for {class_name} to {output_path}")
    
    # Create and save combined grid
    if all_class_images:
        combined_grid = create_grid(all_class_images, grid_size, image_size)
        if combined_grid is not None:
            cv2.putText(combined_grid, "All Classes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add segmentation to filename if appropriate
            prefix = "segmentation_" if is_segmentation else ""
            output_path = os.path.join(output_dir, f"{prefix}all_classes_grid.jpg")
            cv2.imwrite(output_path, combined_grid)
            print(f"Saved combined grid to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create grid visualizations of images")
    parser.add_argument("--input_dir", type=str,
                        default="/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/dataset_2023_built_from_testSet_122/2023_classes/",
                        help="Directory containing class folders with images or a single directory of images")
    parser.add_argument("--output_dir", type=str, 
                        default="/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/dataset_2023_built_from_testSet_122/2023_classes/visualizations",
                        help="Directory to save grid visualizations")
    parser.add_argument("--classes", type=str, nargs='+',
                        help="Space-separated list of class names (subdirectories). If not provided, uses default classes")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to include per class")
    parser.add_argument("--grid_size", type=int, default=10,
                        help="Number of images per row in the grid")
    parser.add_argument("--image_size", type=int, default=100,
                        help="Size to resize each image to (square)")
    parser.add_argument("--single_dir", action="store_true",
                        help="Treat input_dir as a single directory of images, not containing class subdirectories")
    parser.add_argument("--segmentation", action="store_true",
                        help="Indicates that the images are segmentation masks, will enhance visualization")
    
    args = parser.parse_args()
    
    visualize_classes(
        args.input_dir,
        args.output_dir,
        args.classes,
        args.samples,
        args.grid_size,
        (args.image_size, args.image_size),
        args.single_dir,
        args.segmentation
    )
    
    print("Visualization complete!")