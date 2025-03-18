import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from datetime import datetime
import pandas as pd

# Import from your existing module
from NegDataimages import Detector, ImageProcessor, DatasetAnalyzer, Prediction
from Utils import classes, class_colours


class ConfusionMatrixAnalyzer:
    def __init__(self, detector, dataset_analyzer, class_names):
        self.detector = detector
        self.dataset_analyzer = dataset_analyzer
        self.class_names = class_names
        self.true_labels = []
        self.pred_labels = []
        self.img_processor = ImageProcessor(class_names, class_colours)
        
    def process_dataset(self, img_list, label_list, min_area=0.01):
        """
        Process all images in the test set and collect true and predicted labels.
        
        Args:
            img_list: List of paths to images
            label_list: List of paths to label files
            min_area: Minimum area threshold as fraction of image (default: 0.001 or 0.1%)
        """
        for img_path, label_path in zip(img_list, label_list):
            # Get prediction from model
            prediction = self.detector.predict(img_path)
            if not prediction:
                continue
                
            # Extract label information with size filtering
            true_classes = self._extract_true_classes(label_path, min_area)
            
            # Filter predicted masks by size
            pred_classes = []
            results = prediction.results
            
            for r in results:
                mask_data = r.masks.xyn[i]
                class_label = int(r.boxes.cls[i].item())
                
                # Calculate polygon area
                area = self._calculate_polygon_area(mask_data.flatten())
                
                # Only include if area is above threshold
                if area >= min_area:
                    pred_classes.append(class_label)
                else:
                    print(f"Skipping small prediction (class {class_label}, area {area:.6f}) in {os.path.basename(img_path)}")
            
            # Store the class information for each detection
            self.true_labels.extend(true_classes)
            self.pred_labels.extend(pred_classes)
            
            print(f"Processed {os.path.basename(img_path)}: True {true_classes}, Pred {pred_classes}")
    
    def _extract_true_classes(self, label_path, min_area=0.001):
        """
        Extract class labels from ground truth file, filtering by area.
        
        Args:
            label_path: Path to the label file
            min_area: Minimum area threshold as fraction of image (default: 0.001 or 0.1%)
        """
        true_classes = []
        true_areas = []
        
        # Use the ImageProcessor function to load ground truth masks
        ground_truth_masks = self.img_processor.load_ground_truth_masks(label_path)
        
        for mask in ground_truth_masks:
            class_label = int(mask[0])
            coords = mask[1:]
            
            # Calculate polygon area
            area = self._calculate_polygon_area(coords)
            
            # Only include if area is above threshold
            if area >= min_area:
                true_classes.append(class_label)
                true_areas.append(area)
            else:
                print(f"Skipping small object (class {class_label}, area {area:.6f}) in {os.path.basename(label_path)}")
        
        return true_classes

    def _calculate_polygon_area(self, coords):
        """
        Calculate the area of a polygon given its coordinates.
        Coordinates are expected as [x1, y1, x2, y2, ..., xn, yn]
        Area is returned as a fraction of the total image area.
        """
        # Check if coords is empty or has insufficient points (using proper NumPy checks)
        if coords is None or (isinstance(coords, np.ndarray) and coords.size < 6):
            return 0
        
        # Convert to numpy array if it's not already
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        
        # Reshape coordinates to points format [(x1,y1), (x2,y2), ...]
        points = coords.reshape(-1, 2)
        
        # Calculate area using Shoelace formula
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        return area
        
    def build_confusion_matrix(self):
        """Build confusion matrix from collected labels."""
        if not self.true_labels or not self.pred_labels:
            print("No labels collected. Run process_dataset first.")
            return None
        
        # Get the unique classes from both true and predicted labels
        unique_classes = sorted(list(set(self.true_labels + self.pred_labels)))
        
        # Create confusion matrix
        cm = confusion_matrix(
            self.true_labels, 
            self.pred_labels, 
            labels=unique_classes
        )
        
        return cm, unique_classes
    
    def plot_confusion_matrix(self, output_dir, normalize=True):
        """Plot and save confusion matrix."""
        cm, unique_classes = self.build_confusion_matrix()
        if cm is None:
            return
        
        # Create class names for display
        display_names = [self.class_names[i] if i < len(self.class_names) else f"Unknown-{i}" 
                         for i in unique_classes]
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.round(cm, 2)
            title = "Normalized Confusion Matrix"
        else:
            title = "Confusion Matrix"
            
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", 
                    cmap="Blues", xticklabels=display_names, 
                    yticklabels=display_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Create metrics
        metrics = self._calculate_metrics(cm, unique_classes)
            
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{date_str}.png"), dpi=300)
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, f"metrics_{date_str}.csv"))
        
        # Display metrics
        print("\nClassification Metrics:")
        print(metrics_df)
        
        return cm
            
    def _calculate_metrics(self, cm, unique_classes):
        """Calculate precision, recall, and F1 score from confusion matrix."""
        # Initialize metrics dictionary
        metrics = {
            "Class": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": []
        }
        
        # Calculate metrics for each class
        for i, class_idx in enumerate(unique_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            
            # Handle division by zero
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Unknown-{class_idx}"
            
            # Add to metrics
            metrics["Class"].append(class_name)
            metrics["Precision"].append(round(precision, 3))
            metrics["Recall"].append(round(recall, 3))
            metrics["F1 Score"].append(round(f1, 3))
        
        # Calculate global metrics (macro average)
        metrics["Class"].append("AVERAGE")
        metrics["Precision"].append(round(np.mean(metrics["Precision"]), 3))
        metrics["Recall"].append(round(np.mean(metrics["Recall"]), 3))
        metrics["F1 Score"].append(round(np.mean(metrics["F1 Score"]), 3))
        
        return metrics


def main():
    # Configuration
    test_img_folder = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/test_0/labels/images'
    test_label_folder = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/test_0/labels/labels'
    model_weights = "/mnt/hpccs01/home/wardlewo/20250205_cgras_segmentation_alive_dead/train7/weights/best.pt"
    output_dir = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/confusion_matrix_results'
    max_images = 600  # Set max number of images to process
    min_area = 0.01  # Minimum area as fraction of image size (0.1%)
    
    print("Starting Confusion Matrix Analysis")
    
    # Initialize classes
    detector = Detector(model_weights)
    img_processor = ImageProcessor(classes, class_colours)
    dataset_analyzer = DatasetAnalyzer(img_processor)
    
    # Initialize the confusion matrix analyzer
    cm_analyzer = ConfusionMatrixAnalyzer(detector, dataset_analyzer, classes)
    
    # Get list of test images and labels
    img_list = sorted(glob.glob(os.path.join(test_img_folder, '*.jpg')))
    label_list = sorted(glob.glob(os.path.join(test_label_folder, '*.txt')))
    
    # Limit number of images for processing
    img_list = img_list[:max_images]
    label_list = label_list[:max_images]
    
    print(f"Processing {len(img_list)} images with minimum area threshold: {min_area}")
    
    # Process dataset and collect labels
    cm_analyzer.process_dataset(img_list, label_list, min_area)
    
    # Create and plot confusion matrix
    cm_analyzer.plot_confusion_matrix(output_dir, normalize=True)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()