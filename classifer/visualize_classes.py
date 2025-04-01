import os
import random
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import math

# Import classes from segToclassifier.py
classes = ["Alive", "Dead"]

def load_class_images(base_dir, class_name, limit=100):
    """
    Load images for a specific class from its directory
    
    Args:
        base_dir (str): Base directory containing class folders
        class_name (str): Name of the class to load images from
        limit (int): Maximum number of images to load
        
    Returns:
        list: List of loaded images
    """
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        return []
    
    image_files = list(Path(class_dir).glob("*.jpg"))
    
    # Randomly select up to 'limit' images
    if len(image_files) > limit:
        image_files = random.sample(image_files, limit)
    else:
        print(f"Only {len(image_files)} images found for class {class_name}")
    
    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
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

def visualize_classes(input_dir, output_dir, samples_per_class=100, grid_size=10, image_size=(100, 100)):
    """
    Create grid visualizations for each class
    
    Args:
        input_dir (str): Directory containing class folders
        output_dir (str): Directory to save grid images
        samples_per_class (int): Number of samples to include per class
        grid_size (int): Number of images per row in the grid
        image_size (tuple): Size to resize each image to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Also create a combined visualization with samples from all classes
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
            
            output_path = os.path.join(output_dir, f"{class_name}_grid.jpg")
            cv2.imwrite(output_path, grid)
            print(f"Saved grid for {class_name} to {output_path}")
    
    # Create and save combined grid
    if all_class_images:
        combined_grid = create_grid(all_class_images, grid_size, image_size)
        if combined_grid is not None:
            cv2.putText(combined_grid, "All Classes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            output_path = os.path.join(output_dir, "all_classes_grid.jpg")
            cv2.imwrite(output_path, combined_grid)
            print(f"Saved combined grid to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create grid visualizations of classifier images")
    parser.add_argument("--input_dir", type=str, required=False, 
                        help="Directory containing class folders with images")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="Directory to save grid visualizations")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to include per class")
    parser.add_argument("--grid_size", type=int, default=10,
                        help="Number of images per row in the grid")
    parser.add_argument("--image_size", type=int, default=100,
                        help="Size to resize each image to (square)")
    
    args = parser.parse_args()
    
    visualize_classes(
        "/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/dataset_2023_built_from_testSet_122/2023_classes/",
        "/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/dataset_2023_built_from_testSet_122/2023_classes/visualizations",
        args.samples,
        args.grid_size,
        (args.image_size, args.image_size)
    )
    
    print("Visualization complete!")
#python      --samples 100 --grid_size 10 --image_size 100