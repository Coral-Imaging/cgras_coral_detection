import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from ultralytics import YOLO


class SegmentationValidator:
    """
    A class for validating YOLO segmentation models on validation datasets.
    Compares model predictions with ground truth labels and generates
    side-by-side visualizations.
    """
    
    def __init__(self, model_path, yaml_path, output_path, conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize the SegmentationValidator.
        
        Args:
            model_path (str): Path to the YOLO model file (.pt)
            yaml_path (str): Path to the YAML configuration file
            output_path (str): Path to save the validation results
            conf_threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold for validation. Defaults to 0.5.
        """
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.base_output_path = output_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.output_path = self._create_numbered_output_folder(output_path)

        # Load the YOLO model
        self.model = YOLO(model_path)
        self.config = self._load_yaml(yaml_path)
        self.colors = self._generate_colors(len(self.config['names']))

    def _create_numbered_output_folder(self, base_path):
        """Create a new numbered folder for this validation run."""
        # Make sure the base directory exists
        os.makedirs(base_path, exist_ok=True)
        
        # Count existing val_X folders
        existing_dirs = [d for d in os.listdir(base_path) 
                         if os.path.isdir(os.path.join(base_path, d)) and d.startswith('val_')]
        
        # Find the next available number
        run_numbers = [int(d.split('_')[1]) for d in existing_dirs if d.split('_')[1].isdigit()]
        next_number = 1
        if run_numbers:
            next_number = max(run_numbers) + 1
            
        # Create the new directory
        new_dir = os.path.join(base_path, f'val_{next_number}')
        os.makedirs(new_dir, exist_ok=True)
        
        print(f"Creating new validation directory: {new_dir}")
        return new_dir
    
    def _load_yaml(self, yaml_path):
        """Load and parse the YAML configuration file."""
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def _generate_colors(self, num_classes):
        """Generate a distinct color palette for segmentation visualization."""
        base_colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
        return base_colors
    
    def _get_validation_image_paths(self):
        """Get paths to all validation images based on the YAML configuration."""
        base_path = Path(self.config['path'])
        image_paths = []
        
        for val_folder in self.config['val']:
            val_dir = base_path / val_folder
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                image_paths.extend(list(val_dir.glob(f'*{ext}')))
        
        return image_paths
    
    def _get_ground_truth_path(self, img_path):
        """Get the path to the ground truth label file for a given image."""
        # Replace 'images' folder with 'labels' in the path
        label_path = str(img_path).replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        return Path(label_path)
    
    def run_validation(self):
        """Run validation on all validation images and save the visualizations."""
        image_paths = self._get_validation_image_paths()
        
        if not image_paths:
            print("No validation images found. Please check your YAML configuration.")
            return
        
        print(f"Found {len(image_paths)} validation images. Running validation...")
        
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
            self._process_single_image(img_path)
    
    def _process_single_image(self, img_path):
        """Process a single image and save the segmentation visualization."""
        # Run prediction
        results = self.model.predict(
            str(img_path), 
            conf=self.conf_threshold, 
            save=False,
            verbose=False
        )
        
        orig_img = cv2.imread(str(img_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Get ground truth label path
        gt_path = self._get_ground_truth_path(img_path)
        
        if gt_path.exists():
            self._create_comparison_visualization(results[0], orig_img, img_path, gt_path)
        else:
            print(f"Warning: Ground truth labels not found for {img_path}")
            self._create_prediction_only_visualization(results[0], orig_img, img_path)
    
    def _load_ground_truth(self, gt_path, img_shape):
        """Load ground truth labels and convert to mask format."""
        height, width = img_shape[:2]
        gt_masks = []
        gt_classes = []
        
        # YOLO format: class_id, x_center, y_center, width, height
        # Values are normalized to [0, 1]
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) >= 5:  # Basic validation
                        cls_id = int(data[0])
                        
                        # For segmentation in YOLO format, the rest of the line contains polygon coordinates
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
    
    def _create_mask_visualization(self, masks, classes, img_shape):
        """Create a visualization mask from a list of masks and class IDs."""
        height, width = img_shape[:2]
        vis_mask = np.zeros((height, width, 4), dtype=np.float32)
        
        for i, mask in enumerate(masks):
            if i < len(classes):
                cls_id = classes[i]
                color = self.colors[cls_id]
                
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
    
    def _create_comparison_visualization(self, result, orig_img, img_path, gt_path):
        """Create a side-by-side visualization comparing ground truth and predictions."""
        height, width = orig_img.shape[:2]
        
        # Create separate copies of the original image for ground truth and prediction
        gt_img = orig_img.copy()
        pred_img = orig_img.copy()
        
        # Process ground truth
        gt_masks, gt_classes = self._load_ground_truth(gt_path, gt_img.shape)
        gt_vis_mask = self._create_mask_visualization(gt_masks, gt_classes, gt_img.shape)
        
        # Process predictions
        pred_masks = []
        pred_classes = []
        
        if result.masks is not None and len(result.masks) > 0:
            for i, mask in enumerate(result.masks.data):
                cls_id = int(result.boxes.cls[i].item())
                conf = result.boxes.conf[i].item()
                
                if conf < self.conf_threshold:
                    continue
                
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (width, height))
                pred_masks.append(mask_np)
                pred_classes.append(cls_id)
        
        pred_vis_mask = self._create_mask_visualization(pred_masks, pred_classes, pred_img.shape)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Ground truth visualization
        ax1.imshow(gt_img)
        ax1.imshow(gt_vis_mask, alpha=0.5)
        ax1.set_title('Ground Truth')
        ax1.axis('off')
        
        # Prediction visualization
        ax2.imshow(pred_img)
        ax2.imshow(pred_vis_mask, alpha=0.5)
        ax2.set_title('Model Prediction')
        ax2.axis('off')
        
        # Create legend
        legend_elements = []
        for class_id, class_name in self.config['names'].items():
            color = self.colors[class_id]
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=color, label=f'{class_name}')
            )
        
        fig.legend(
            handles=legend_elements, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.05), 
            ncol=min(5, len(legend_elements))
        )
        
        fig.suptitle(f'Segmentation Validation - {Path(img_path).name}', fontsize=16)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Adjust for title and legend
        
        output_file = Path(self.output_path) / f"{Path(img_path).stem}_val.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved validation visualization to {output_file}")
    
    def _create_prediction_only_visualization(self, result, orig_img, img_path):
        """Create a visualization with prediction only when ground truth is not available."""
        height, width = orig_img.shape[:2]
        pred_mask = np.zeros((height, width, 4), dtype=np.float32)
        
        if result.masks is not None and len(result.masks) > 0:
            for i, mask in enumerate(result.masks.data):
                cls_id = int(result.boxes.cls[i].item())
                conf = result.boxes.conf[i].item()
                
                if conf < self.conf_threshold:
                    continue
                
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (width, height))
                
                color = self.colors[cls_id]
                
                for c in range(3):
                    pred_mask[..., c] = np.where(
                        mask_np > 0.5,
                        color[c] * 0.7 + pred_mask[..., c] * 0.3,
                        pred_mask[..., c]
                    )
                
                pred_mask[..., 3] = np.where(
                    mask_np > 0.5,
                    0.7,  # Semi-transparent
                    pred_mask[..., 3]
                )
        
        plt.figure(figsize=(12, 8))
        plt.imshow(orig_img)
        plt.imshow(pred_mask, alpha=0.5)
        
        legend_elements = []
        for class_id, class_name in self.config['names'].items():
            color = self.colors[class_id]
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=color, label=f'{class_name}')
            )
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.title(f'Segmentation Prediction (No Ground Truth) - {Path(img_path).name}')
        plt.axis('off')
        
        output_file = Path(self.output_path) / f"{Path(img_path).stem}_prediction.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved prediction visualization to {output_file}")
    
    def generate_summary(self):
        """Generate a summary of the validation results."""
        print(f"Segmentation validation completed. Results saved to {self.output_path}")
        
        output_files = list(Path(self.output_path).glob("*_validation.png"))
        output_files.extend(list(Path(self.output_path).glob("*_prediction.png")))
        print(f"Processed {len(output_files)} images")