#!/usr/bin/env python3
"""
SAHI Image Tiling and Prediction Script

This script processes a folder of images by:
1. Tiling them using SAHI
2. Running predictions on each tile with a specified model
3. Stitching the predictions back together
4. Optionally exporting results in CVAT-compatible format

Usage:
    python sahi_tile_predict.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import click
import cv2
import numpy as np
from pathlib import Path
import shutil
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction


@click.command()
@click.option('--input_dir', required=True, help='Directory containing input images')
@click.option('--output_dir', default='./output', help='Directory to save output')
@click.option('--model_path', default="/home/java/hpc-home/cgras_segmentation/train_coral_polyp/weights/best.pt", 
              help='Path to the model weights')
@click.option('--model_type', default='ultralytics', help='Model type (ultralytics, mmdet, etc.)')
@click.option('--device', default='cpu', help='Device to run inference on (cpu/cuda/cuda:0)')
@click.option('--slice_height', default=640, help='Height of each slice')
@click.option('--slice_width', default=640, help='Width of each slice')
@click.option('--overlap', default=0.3, help='Overlap ratio between slices')
@click.option('--conf_thresh', default=0.5, help='Confidence threshold for predictions')
@click.option('--export_cvat', is_flag=True, help='Export results in CVAT-compatible format')
@click.option('--visualize', is_flag=True, help='Generate visualization of predictions')
@click.option('--keep_slices', is_flag=True, help='Keep intermediate slice files')
def main(input_dir, output_dir, model_path, model_type, device, 
         slice_height, slice_width, overlap, conf_thresh, 
         export_cvat, visualize, keep_slices):
    """Process images with SAHI tiling and prediction."""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    slices_dir = os.path.join(output_dir, 'slices')
    predictions_dir = os.path.join(output_dir, 'predictions')
    cvat_dir = os.path.join(output_dir, 'cvat_annotations') if export_cvat else None
    
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    if cvat_dir:
        os.makedirs(cvat_dir, exist_ok=True)
    
    # Initialize model
    click.echo(f"Loading model from {model_path}...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=conf_thresh,
        device=device
    )
    
    # Process each image in input directory
    image_paths = sorted([p for p in Path(input_dir).glob('**/*') 
                         if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']])
    
    if not image_paths:
        click.echo(f"No images found in {input_dir}")
        return
    
    click.echo(f"Found {len(image_paths)} images to process")
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_name = img_path.name
        img_stem = img_path.stem
        
        click.echo(f"\nProcessing {img_name}")
        
        # Create image-specific output directories
        img_slices_dir = os.path.join(slices_dir, img_stem)
        os.makedirs(img_slices_dir, exist_ok=True)
        
        # Get sliced predictions
        sliced_results = get_sliced_prediction(
            image=str(img_path),
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type='NMM',  # Non-Maximum Merging
            slice_dir=img_slices_dir,
            slice_export_prefix='slice',
            verbose=False
        )
        
        # Export predictions
        prediction_path = os.path.join(predictions_dir, f"{img_stem}_prediction.jpg")
        sliced_results.export_visuals(
            export_dir=predictions_dir,
            file_name=img_stem,
            hide_conf=False,
            text_size=1,
            rect_th=2
        )
        
        # Export to CVAT format if requested
        if export_cvat:
            export_to_cvat(sliced_results, img_path, cvat_dir)
        
        # Visualize individual slices if requested
        if visualize:
            visualize_slices(img_slices_dir, detection_model, 
                            os.path.join(output_dir, 'visualizations', img_stem))
        
        # Clean up intermediate files if not keeping them
        if not keep_slices:
            shutil.rmtree(img_slices_dir)
    
    click.echo("Processing complete!")


def visualize_slices(slices_dir: str, detection_model, output_dir: str) -> None:
    """Visualize predictions on individual slices and create a grid visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all slice images
    slice_paths = sorted(Path(slices_dir).glob('*.png'))
    if not slice_paths:
        return
    
    # Process each slice
    slice_predictions = []
    for slice_path in slice_paths:
        # Get prediction for this slice
        pred = get_prediction(str(slice_path), detection_model)
        
        # Export visual
        output_name = slice_path.stem
        pred.export_visuals(
            export_dir=output_dir,
            file_name=output_name,
            hide_conf=False,
            text_size=0.6,
            rect_th=2
        )
        
        # Store prediction info
        pred_path = os.path.join(output_dir, f"{output_name}.png")
        coords = extract_coords_from_filename(slice_path.name)
        if coords:
            slice_predictions.append((pred_path, coords))
    
    # Create grid visualization
    create_grid_visualization(slice_predictions, os.path.join(output_dir, "grid_visualization.jpg"))


def extract_coords_from_filename(filename: str) -> Optional[Tuple[int, int, int, int]]:
    """Extract coordinates from a slice filename."""
    parts = filename.replace(".png", "").split("_")
    if len(parts) >= 5:  # Expected format: slice_x1_y1_x2_y2
        try:
            x1, y1, x2, y2 = map(int, parts[1:5])
            return (x1, y1, x2, y2)
        except (ValueError, IndexError):
            pass
    return None


def create_grid_visualization(slices: List[Tuple[str, Tuple[int, int, int, int]]], output_path: str) -> None:
    """Create a grid visualization of slices."""
    if not slices:
        return
    
    # Sort slices by (y1, x1) so they appear in correct order
    slices.sort(key=lambda x: (x[1][1], x[1][0]))
    
    # Load images
    images = [cv2.imread(filename) for filename, _ in slices]
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(images))))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    # Plot each slice
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].axis("off")
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_to_cvat(results, image_path: Path, cvat_dir: str) -> None:
    """Export results to CVAT-compatible format."""
    image_name = image_path.name
    
    # Create CVAT annotation structure
    cvat_annotations = {
        "version": 1,
        "tags": [],
        "shapes": [],
        "tracks": []
    }
    
    # Add each prediction as a bounding box
    for pred in results.object_prediction_list:
        bbox = pred.bbox.to_voc_bbox()  # [xmin, ymin, xmax, ymax]
        score = pred.score.value
        category_name = pred.category.name
        
        shape = {
            "type": "rectangle",
            "frame": 0,
            "label": category_name,
            "points": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "occluded": False,
            "attributes": {
                "confidence": score
            }
        }
        
        cvat_annotations["shapes"].append(shape)
    
    # Save annotations to JSON file
    annotation_path = os.path.join(cvat_dir, f"{image_path.stem}.json")
    with open(annotation_path, 'w') as f:
        json.dump(cvat_annotations, f, indent=2)


if __name__ == "__main__":
    main()