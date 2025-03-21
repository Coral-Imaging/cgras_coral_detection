#!/usr/bin/env python3
"""
A standalone script to visualize YOLOv8 segmentation predictions on a single image.
Creates a detailed subplot visualization showing:
1. Original image
2. Ground truth segmentation
3. All predictions combined
4. Individual prediction masks
"""

import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from ultralytics import YOLO

# Hardcoded parameters - change these as needed
MODEL_PATH = "/home/java/hpc-home/cgras_segmentation/train_coral_polyp2/weights/best.pt"

YAML_PATH = "/media/java/RRAP03/outputs/train/cgras_data.yaml"
# IMAGE_PATH = "/media/java/RRAP03/outputs/train/split_dataset/export_cgras_2024_amag_T01_first10_100quality/valid/images/CGRAS_Amag_MIS5a_20241211_w6_T01_07_320_4160.jpg"
IMAGE_PATH = "/media/java/RRAP03/outputs/train/split_dataset/export_cgras_2024_amag_T01_first10_100quality/valid/images/CGRAS_Amag_MIS5a_20241105_w1_T01_13_7040_1600.jpg"
OUTPUT_DIR = "/media/java/RRAP03/outputs/test_image/"
CONF_THRESHOLD = 0.5

def generate_unique_filename(output_dir, prefix="output", extension=".png"):
    """Generate a unique filename in the given directory using a timestamp."""
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp for unique filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with timestamp
    filename = f"{prefix}_{timestamp}{extension}"
    return os.path.join(output_dir, filename)

def load_yaml(yaml_path):
    """Load and parse the YAML configuration file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_ground_truth_path(img_path):
    """Get the path to the ground truth label file for a given image."""
    # Replace 'images' folder with 'labels' in the path and change extension to .txt
    label_path = str(img_path).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
    return Path(label_path)

def load_ground_truth(gt_path, img_shape):
    """Load ground truth labels and convert to mask format."""
    height, width = img_shape[:2]
    gt_masks = []
    gt_classes = []
    
    # YOLO format: class_id, x_center, y_center, width, height
    # Or for segmentation: class_id, x1, y1, x2, y2, ...
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) >= 5:  # Basic validation
                    cls_id = int(data[0])
                    
                    # For segmentation in YOLO format, check if we have polygon coordinates
                    if len(data) > 5:  # This is a segmentation mask
                        # Convert polygon coordinates from normalized to absolute
                        polygon = []
                        for i in range(1, len(data), 2):
                            if i+1 < len(data):
                                x = float(data[i]) * width
                                y = float(data[i+1]) * height
                                polygon.append([x, y])
                        
                        # Create a mask from polygon
                        mask = np.zeros((height, width), dtype=np.uint8)
                        if len(polygon) >= 3:  # Need at least 3 points for a polygon
                            polygon_np = np.array(polygon, dtype=np.int32)
                            cv2.fillPoly(mask, [polygon_np], 1)
                            gt_masks.append(mask)
                            gt_classes.append(cls_id)
                    else:  # This is a bounding box
                        x_center = float(data[1]) * width
                        y_center = float(data[2]) * height
                        box_width = float(data[3]) * width
                        box_height = float(data[4]) * height
                        
                        # Convert to top-left and bottom-right coordinates
                        x1 = int(x_center - box_width / 2)
                        y1 = int(y_center - box_height / 2)
                        x2 = int(x_center + box_width / 2)
                        y2 = int(y_center + box_height / 2)
                        
                        # Create a mask from bounding box
                        mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=cv2.FILLED)
                        gt_masks.append(mask)
                        gt_classes.append(cls_id)
    
    return gt_masks, gt_classes

def create_mask_visualization(masks, classes, img_shape, colors):
    """Create a visualization mask from a list of masks and class IDs."""
    height, width = img_shape[:2]
    vis_mask = np.zeros((height, width, 4), dtype=np.float32)
    
    for i, mask in enumerate(masks):
        if i < len(classes):
            cls_id = classes[i]
            color = colors[cls_id]
            
            for c in range(3):
                vis_mask[..., c] = np.where(
                    mask > 0,
                    color[c] * 0.7 + vis_mask[..., c] * 0.3,
                    vis_mask[..., c]
                )
            
            vis_mask[..., 3] = np.where(
                mask > 0,
                0.7,  # Semi-transparent
                vis_mask[..., 3]
            )
    
    return vis_mask

def visualize_segmentation():
    """Main function to create segmentation visualization."""
    # Load model and configuration
    output_path = generate_unique_filename(OUTPUT_DIR)
    model = YOLO(MODEL_PATH)
    config = load_yaml(YAML_PATH)
    
    # Generate colors for visualization
    num_classes = len(config['names'])
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    # Load and process the image
    img_path = Path(IMAGE_PATH)
    orig_img = cv2.imread(str(img_path))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    height, width = orig_img.shape[:2]
    
    # Run prediction
    results = model.predict(
        str(img_path),
        conf=CONF_THRESHOLD,
        save=False,
        verbose=False
    )
    result = results[0]
    
    # Get ground truth
    gt_path = get_ground_truth_path(img_path)
    gt_masks, gt_classes = load_ground_truth(gt_path, orig_img.shape)
    
    # Extract prediction masks and classes
    pred_masks = []
    pred_classes = []
    
    print("\n=== SEGMENTATION PREDICTIONS ===")
    print(f"{'Class':<15} {'Confidence':<12} {'X-Center':<10} {'Y-Center':<10} {'Width':<10} {'Height':<10}")
    print("-" * 70)
    
    if result.masks is not None and len(result.masks) > 0:
        for i, mask in enumerate(result.masks.data):
            cls_id = int(result.boxes.cls[i].item())
            conf = result.boxes.conf[i].item()
            
            if conf < CONF_THRESHOLD:
                continue
            
            # Get the bounding box coordinates
            box = result.boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Calculate center and dimensions
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Get mask centroid as alternative location info
            mask_np = mask.cpu().numpy()
            mask_np = cv2.resize(mask_np, (width, height))
            
            # Find contours in the mask
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate mask centroid if contours exist
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x_center, y_center  # Fallback to box center
            else:
                cx, cy = x_center, y_center  # Fallback to box center
            
            # Print prediction info to console
            class_name = config['names'][cls_id]
            print(f"{class_name:<15} {conf:.6f}    {cx:<10.1f} {cy:<10.1f} {box_width:<10.1f} {box_height:<10.1f}")
            
            # Store for visualization
            pred_masks.append(mask_np)
            pred_classes.append(cls_id)
    
    print("-" * 70)
    print(f"Total predictions: {len(pred_masks)}\n")
    
    # Create visualizations
    gt_vis_mask = create_mask_visualization(gt_masks, gt_classes, orig_img.shape, colors)
    pred_vis_mask = create_mask_visualization(pred_masks, pred_classes, orig_img.shape, colors)
    
    # Create a figure with subplots
    num_pred = len(pred_masks)  # Show all predictions
    
    # Calculate grid dimensions based on number of predictions
    if num_pred <= 3:
        # With few predictions, use a simple layout
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, max(3, num_pred))
        
        # Top row: Original, GT, All predictions
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(orig_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(orig_img)
        ax2.imshow(gt_vis_mask, alpha=0.5)
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(orig_img)
        ax3.imshow(pred_vis_mask, alpha=0.5)
        ax3.set_title('All Predictions')
        ax3.axis('off')
        
        # Bottom two rows: Individual predictions
        for i in range(num_pred):
            if i < len(pred_masks):
                mask = pred_masks[i]
                cls_id = pred_classes[i]
                
                # For individual mask visualization
                ind_mask = np.zeros((height, width, 4), dtype=np.float32)
                color = colors[cls_id]
                
                for c in range(3):
                    ind_mask[..., c] = np.where(mask > 0.5, color[c], 0)
                
                ind_mask[..., 3] = np.where(mask > 0.5, 0.7, 0)
                
                ax_ind = fig.add_subplot(gs[1:, i])
                ax_ind.imshow(orig_img)
                ax_ind.imshow(ind_mask, alpha=0.5)
                class_name = config['names'][cls_id]
                conf = result.boxes.conf[i].item() if i < len(result.boxes) else 0
                ax_ind.set_title(f'Class: {class_name} (Conf: {conf:.2f})')
                ax_ind.axis('off')
                
    else:
        # With many predictions, use a more complex layout
        # First, determine a reasonable grid size
        num_cols = min(4, num_pred)  # Max 4 columns
        num_rows = (num_pred + num_cols - 1) // num_cols + 1  # +1 for the header row
        
        fig = plt.figure(figsize=(20, 5 * num_rows))
        
        # Create a separate gridspec for the top row
        gs_top = fig.add_gridspec(1, 3, height_ratios=[1], top=0.95, bottom=0.75)
        
        # Top row: Original, GT, All predictions (spans 1 row, 3 columns)
        ax1 = fig.add_subplot(gs_top[0, 0])
        ax1.imshow(orig_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs_top[0, 1])
        ax2.imshow(orig_img)
        ax2.imshow(gt_vis_mask, alpha=0.5)
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs_top[0, 2])
        ax3.imshow(orig_img)
        ax3.imshow(pred_vis_mask, alpha=0.5)
        ax3.set_title('All Predictions')
        ax3.axis('off')
        
        # Create a separate gridspec for individual predictions
        gs_preds = fig.add_gridspec(num_rows-1, num_cols, top=0.70, bottom=0.1)
        
        # Individual prediction masks
        for i in range(num_pred):
            if i < len(pred_masks):
                # Calculate row and column position in the grid
                row = i // num_cols
                col = i % num_cols
                
                mask = pred_masks[i]
                cls_id = pred_classes[i]
                
                # For individual mask visualization
                ind_mask = np.zeros((height, width, 4), dtype=np.float32)
                color = colors[cls_id]
                
                for c in range(3):
                    ind_mask[..., c] = np.where(mask > 0.5, color[c], 0)
                
                ind_mask[..., 3] = np.where(mask > 0.5, 0.7, 0)
                
                ax_ind = fig.add_subplot(gs_preds[row, col])
                ax_ind.imshow(orig_img)
                ax_ind.imshow(ind_mask, alpha=0.5)
                class_name = config['names'][cls_id]
                conf = result.boxes.conf[i].item() if i < len(result.boxes) else 0
                ax_ind.set_title(f'Class: {class_name} (Conf: {conf:.2f})')
                ax_ind.axis('off')
    
    # Create legend
    legend_elements = []
    for class_id, class_name in config['names'].items():
        color = colors[class_id]
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color=color, label=f'{class_name}')
        )
    
    # Add legend at the bottom of the figure
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=min(5, len(legend_elements))
    )
    
    # Add overall title
    fig.suptitle(f'Segmentation Analysis - {Path(img_path).name}', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjust for title and legend
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved segmentation visualization to {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_segmentation()
