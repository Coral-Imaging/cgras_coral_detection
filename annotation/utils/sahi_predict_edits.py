#!/usr/bin/env python3
"""
SAHI prediction script with COCO format output wrapped in a class structure.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from PIL import Image
from sahi import AutoDetectionModel
from sahi.prediction import PredictionResult
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation

# Import SAHI components
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import list_files_with_extensions


class SahiPredictor:
    """
    A class to run SAHI predictions and generate COCO format annotations.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the SahiPredictor.
        
        Args:
            verbose (bool): Whether to print progress messages
        """
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.detection_model = None
        self.categories = ["alive_coral", "dead_coral"]
        self.category_mapping = {}
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def initialize_model(self, model_path, conf_thresh=0.4, device="cuda:0"):
        """
        Initialize the detection model.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            conf_thresh (float): Confidence threshold
            device (str): Device to run inference on
        """
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=conf_thresh,
            device=device
        )
        if self.verbose:
            self.logger.info(f"Model initialized from {model_path}")
    
    def setup_coco_structure(self):
        """
        Initialize COCO object with categories.
        
        Returns:
            Coco: Initialized COCO object
        """
        coco = Coco()
        
        # Define categories and mapping
        for i, category_name in enumerate(self.categories, 1):
            category = CocoCategory(id=i, name=category_name)
            coco.add_category(category)
            self.category_mapping[category_name] = i
        
        return coco
    
    def get_image_files(self, data_dir):
        """
        Get list of image files from directory.
        
        Args:
            data_dir (str): Directory containing images
            
        Returns:
            list: List of image file paths
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        image_files = list_files_with_extensions(
            directory=data_dir,
            extensions=[".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        )
        
        if self.verbose:
            self.logger.info(f"Found {len(image_files)} images in {data_dir}")
        
        return image_files
    
    def predict_image(self, image_path, slice_width=640, slice_height=640, overlap=0.5):
        """
        Run prediction on a single image.
        
        Args:
            image_path (str): Path to image file
            slice_width (int): Width of slices
            slice_height (int): Height of slices
            overlap (float): Overlap ratio for slicing
            
        Returns:
            PredictionResult: Prediction result object
        """
        try:
            # Try sliced prediction with error handling
            result = get_sliced_prediction(
                image=image_path,
                detection_model=self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                perform_standard_pred=False,
                postprocess_type="NMS",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.2,
                verbose=0
            )
        except ValueError as e:
            self.logger.warning(f"Sliced prediction with NMS failed for {os.path.basename(image_path)}: {e}")
            try:
                # Try direct prediction as fallback
                detection_result = self.detection_model.predict(image=image_path)
                result = PredictionResult(
                    image=image_path,
                    object_prediction_list=detection_result.object_prediction_list
                )
                if self.verbose:
                    self.logger.info(f"Used direct prediction for {os.path.basename(image_path)}")
            except Exception as direct_err:
                self.logger.error(f"Direct prediction failed for {os.path.basename(image_path)}: {direct_err}")
                # Create empty result to continue processing
                result = PredictionResult(image=image_path, object_prediction_list=[])
        
        return result
    
    def create_coco_annotation(self, pred, coco_image, annotation_id):
        """
        Create a COCO annotation from a prediction.
        
        Args:
            pred: Prediction object
            coco_image: COCO image object
            annotation_id (int): Annotation ID
            
        Returns:
            tuple: (CocoAnnotation or None, next_annotation_id)
        """
        try:
            # Get category info
            predicted_category_name = pred.category.name
            if predicted_category_name in self.category_mapping:
                category_id = self.category_mapping[predicted_category_name]
            else:
                self.logger.warning(f"Unknown category '{predicted_category_name}', using default")
                category_id = 1
                predicted_category_name = self.categories[0]

            # Get bounding box in COCO format [x, y, width, height]
            x = pred.bbox.minx
            y = pred.bbox.miny
            box_width = pred.bbox.maxx - pred.bbox.minx
            box_height = pred.bbox.maxy - pred.bbox.miny
            
            # Skip annotations with zero width or height
            if box_width <= 0 or box_height <= 0:
                self.logger.warning(f"Skipping invalid bounding box with zero width/height")
                return None, annotation_id
                
            bbox = [x, y, box_width, box_height]
            area = box_width * box_height
            
            # Handle segmentation safely
            segmentation = self._get_segmentation(pred, x, y, box_width, box_height)
            if not segmentation:
                return None, annotation_id
            
            # Create COCO annotation
            coco_annotation = CocoAnnotation(
                bbox=bbox,
                category_id=category_id,
                category_name=predicted_category_name,
                image_id=coco_image.id,
                segmentation=segmentation,
                iscrowd=0
            )
            
            # Set ID and area manually after creation
            coco_annotation.id = annotation_id
            coco_annotation.json["area"] = area
            
            return coco_annotation, annotation_id + 1
            
        except Exception as e:
            self.logger.error(f"Error creating annotation: {e}")
            return None, annotation_id
    
    def _get_segmentation(self, pred, x, y, box_width, box_height):
        """
        Get segmentation from prediction or create from bbox as fallback.
        
        Returns:
            list: Segmentation coordinates or None if invalid
        """
        try:
            if hasattr(pred, 'mask') and pred.mask is not None and hasattr(pred.mask, 'segmentation'):
                segmentation = pred.mask.segmentation
                # Validate segmentation format
                if not isinstance(segmentation, list) or len(segmentation) == 0:
                    raise ValueError("Empty segmentation list")
            else:
                raise AttributeError("No valid mask found")
        except (AttributeError, ValueError) as e:
            # Create segmentation from bbox as fallback
            self.logger.debug(f"Using bbox for segmentation due to: {e}")
            segmentation = [[
                x, y, 
                x + box_width, y,
                x + box_width, y + box_height,
                x, y + box_height
            ]]
        
        # Validate segmentation
        if not segmentation or not all(isinstance(s, list) and len(s) > 0 for s in segmentation):
            self.logger.warning(f"Invalid segmentation format")
            return None
            
        return segmentation
    
    def process_images(self, image_files, slice_width=640, slice_height=640, overlap=0.5):
        """
        Process all images and create COCO annotations.
        
        Args:
            image_files (list): List of image file paths
            slice_width (int): Width of slices
            slice_height (int): Height of slices
            overlap (float): Overlap ratio for slicing
            
        Returns:
            Coco: COCO object with all annotations
        """
        coco = self.setup_coco_structure()
        annotation_id = 1
        
        for image_path in tqdm(image_files, desc="Processing images", disable=not self.verbose):
            try:
                # Get image info
                image_filename = os.path.basename(image_path)
                image = Image.open(image_path)
                width, height = image.size
                
                # Create COCO image
                coco_image = CocoImage(
                    file_name=image_filename,
                    height=height,
                    width=width,
                )
                
                # Get predictions
                result = self.predict_image(image_path, slice_width, slice_height, overlap)
                
                # Process each prediction
                for pred in result.object_prediction_list:
                    coco_annotation, annotation_id = self.create_coco_annotation(
                        pred, coco_image, annotation_id
                    )
                    if coco_annotation is not None:
                        coco_image.add_annotation(coco_annotation)
                
                # Add the image to COCO object
                coco.add_image(coco_image)
                
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        return coco
    
    def finalize_coco_json(self, coco, name="default"):
        """
        Finalize COCO JSON with proper info and licenses sections.
        
        Args:
            coco (Coco): COCO object
            name (str): Experiment name
            
        Returns:
            dict: Complete COCO JSON dictionary
        """
        coco_json = coco.json
        coco_json["info"] = {
            "description": f"COCO-formatted annotations for {name}",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "SAHI automatic annotation",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        coco_json["licenses"] = [
            {
                "url": "",
                "id": 1,
                "name": "Unknown"
            }
        ]
        
        return coco_json
    
    def save_coco_json(self, coco_json, output_path):
        """
        Save COCO JSON to file.
        
        Args:
            coco_json (dict): COCO JSON dictionary
            output_path (str): Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coco_json, f)
        
        if self.verbose:
            self.logger.info(f"COCO annotations saved to: {output_path}")
    
    def run_prediction_pipeline(self, data_dir, output_dir, model_path, name="default",
                               slice_width=640, slice_height=640, overlap=0.5,
                               conf_thresh=0.4, device="cuda:0"):
        """
        Complete prediction pipeline from images to COCO JSON.
        
        Args:
            data_dir (str): Directory containing images
            output_dir (str): Output directory for annotations
            model_path (str): Path to YOLOv8 model weights
            name (str): Experiment name
            slice_width (int): Width of slices
            slice_height (int): Height of slices
            overlap (float): Overlap ratio for slicing
            conf_thresh (float): Confidence threshold
            device (str): Device to run inference on
            
        Returns:
            str: Path to saved COCO JSON file
        """
        # Initialize model
        self.initialize_model(model_path, conf_thresh, device)
        
        # Get image files
        image_files = self.get_image_files(data_dir)
        
        # Create output directory
        annotations_dir = os.path.join(
            output_dir, 
            f"{name}_annotations_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_coco 1.0"
        )
        
        # Process images
        coco = self.process_images(image_files, slice_width, slice_height, overlap)
        
        # Finalize and save
        coco_json = self.finalize_coco_json(coco, name)
        output_path = os.path.join(annotations_dir, "annotations", "instances_default.json")
        self.save_coco_json(coco_json, output_path)
        
        if self.verbose:
            self.logger.info(f"Total images processed: {len(coco.images)}")
        
        return output_path


# Legacy functions for backward compatibility
def parse_arguments():
    """Legacy argument parser for backward compatibility."""
    parser = argparse.ArgumentParser(description="Run SAHI prediction with COCO format output")
    parser.add_argument("--project", type=str, default="sahi_project", help="Project name")
    parser.add_argument("--name", type=str, default="default", help="Experiment name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for annotations")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLOv8 model weights")
    parser.add_argument("--slice_width", type=int, default=640, help="Width of slices")
    parser.add_argument("--slice_height", type=int, default=640, help="Height of slices")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio for slicing")
    parser.add_argument("--conf_thresh", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    return parser.parse_args()


def main():
    """Main function using the class structure."""
    # Parse arguments or use defaults
    try:
        args = parse_arguments()
        predictor = SahiPredictor(verbose=True)
        predictor.run_prediction_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            name=args.name,
            slice_width=args.slice_width,
            slice_height=args.slice_height,
            overlap=args.overlap,
            conf_thresh=args.conf_thresh,
            device=args.device
        )
    except:
        # Use defaults if run without argparse
        predictor = SahiPredictor(verbose=True)
        predictor.run_prediction_pipeline(
            data_dir="/media/wardlewo/RRAP02/cgras_pdae_2024_aims/pdae_100",
            output_dir="/home/wardlewo/Reggie/data",
            model_path="/home/wardlewo/hpc-home/runs/pdae_29042025_cgras_seg_first_30/20250326_8n_train_multiGpu_B128/weights/best.pt",
            name="Pdae_100"
        )


if __name__ == "__main__":
    main()