import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from ultralytics import YOLO


class SegmentationTester:
    """
    A class for testing YOLO segmentation models on test datasets.
    Generates segmentation visualizations with class legends.
    """
    
    def __init__(self, model_path, yaml_path, output_path, conf_threshold=0.5):
        """
        Initialize the SegmentationTester.
        
        Args:
            model_path (str): Path to the YOLO model file (.pt)
            yaml_path (str): Path to the YAML configuration file
            output_path (str): Path to save the segmentation results
            conf_threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
        """
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.base_output_path = output_path
        self.conf_threshold = conf_threshold

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
                         if os.path.isdir(os.path.join(base_path, d)) and d.startswith('test_')]
        
        # Find the next available number
        run_numbers = [int(d.split('_')[1]) for d in existing_dirs if d.split('_')[1].isdigit()]
        next_number = 1
        if run_numbers:
            next_number = max(run_numbers) + 1
            
        # Create the new directory
        new_dir = os.path.join(base_path, f'test_{next_number}')
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
    
    def _get_test_image_paths(self):
        """Get paths to all test images based on the YAML configuration."""
        base_path = Path(self.config['path'])
        image_paths = []
        
        for test_folder in self.config['test']:
            test_dir = base_path / test_folder
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                image_paths.extend(list(test_dir.glob(f'*{ext}')))
        
        return image_paths
    
    def run_predictions(self):
        """Run predictions on all test images and save the visualizations."""
        image_paths = self._get_test_image_paths()
        
        if not image_paths:
            print("No test images found. Please check your YAML configuration.")
            return
        
        print(f"Found {len(image_paths)} test images. Running predictions...")
        
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
        
        self._create_visualization(results[0], orig_img, img_path)
    
    def _create_visualization(self, result, orig_img, img_path):
        """Create a visualization of the segmentation results with a legend."""
        height, width = orig_img.shape[:2]
        segmentation_mask = np.zeros((height, width, 4), dtype=np.float32)
        
        if result.masks is None or len(result.masks) == 0:
            print(f"No segmentation masks found for {img_path}")
        else:
            for i, mask in enumerate(result.masks.data):
                cls_id = int(result.boxes.cls[i].item())
                conf = result.boxes.conf[i].item()
                
                if conf < self.conf_threshold:
                    continue
                
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (width, height))
                
                color = self.colors[cls_id]
                
                for c in range(3):
                    segmentation_mask[..., c] = np.where(
                        mask_np > 0.5,
                        color[c] * 0.7 + segmentation_mask[..., c] * 0.3,
                        segmentation_mask[..., c]
                    )

                segmentation_mask[..., 3] = np.where(
                    mask_np > 0.5,
                    0.7,  # Semi-transparent
                    segmentation_mask[..., 3]
                )
        
        plt.figure(figsize=(12, 8))
        plt.imshow(orig_img)
        plt.imshow(segmentation_mask, alpha=0.5)
        
        legend_elements = []
        for class_id, class_name in self.config['names'].items():
            color = self.colors[class_id]
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=color, label=f'{class_name}')
            )
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.title(f'Segmentation Results - {Path(img_path).name}')
        plt.axis('off')
        
        output_file = Path(self.output_path) / f"{Path(img_path).stem}_segmented.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved segmentation visualization to {output_file}")
    
    def generate_summary(self):
        """Generate a summary of the test results."""
        print(f"Segmentation testing completed. Results saved to {self.output_path}")
        
        output_files = list(Path(self.output_path).glob("*_segmented.png"))
        print(f"Processed {len(output_files)} images")