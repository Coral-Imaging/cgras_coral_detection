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
        self.processed_images = 0
        self.skipped_images = 0
        # Add background class index as last class index + 1
        self.background_idx = len(class_names)
        
    def process_dataset(self, img_list, label_list, min_area=0.01):
        """
        Process all images in the test set and collect true and predicted labels.
        Include background class for false positives/negatives.
        
        Args:
            img_list: List of paths to images
            label_list: List of paths to label files
            min_area: Minimum area threshold as fraction of image (default: 0.01 or 1%)
        """
        for img_path, label_path in zip(img_list, label_list):
            try:
                # Get prediction from model
                prediction = self.detector.predict(img_path)
                if not prediction:
                    print(f"No prediction for {os.path.basename(img_path)}, skipping")
                    self.skipped_images += 1
                    continue
                    
                # Extract label information with size filtering
                true_classes = self._extract_true_classes(label_path, min_area)
                
                # Filter predicted masks by size
                pred_classes = []
                pred_masks = []  # Store masks for IOU calculation
                results = prediction.results
                
                for r in results:
                    # Check if we have masks available
                    if hasattr(r, 'masks') and r.masks is not None and len(r.masks.xyn) > 0:
                        # Iterate through each mask in the current result
                        for i, mask_data in enumerate(r.masks.xyn):
                            class_label = int(r.boxes.cls[i].item())
                            
                            # Calculate polygon area
                            area = self._calculate_polygon_area(mask_data.flatten())
                            
                            # Only include if area is above threshold
                            if area >= min_area:
                                pred_classes.append(class_label)
                                pred_masks.append(mask_data.flatten())
                            else:
                                print(f"Skipping small prediction (class {class_label}, area {area:.6f}) in {os.path.basename(img_path)}")
                    else:
                        # Handle case where masks are not available but boxes are
                        for i in range(len(r.boxes)):
                            class_label = int(r.boxes.cls[i].item())
                            pred_classes.append(class_label)
                            # No mask available, use empty mask
                            pred_masks.append(None)
                
                # Load ground truth masks for IOU calculation
                true_mask_data = self.img_processor.load_ground_truth_masks(label_path)
                true_masks = []
                for mask in true_mask_data:
                    if min_area <= self._calculate_polygon_area(mask[1:]):
                        true_masks.append(mask[1:])
                
                # If we have predictions but no ground truth
                if pred_classes and not true_classes:
                    # All predictions are false positives (background)
                    for pred_class in pred_classes:
                        self.true_labels.append(self.background_idx)  # Background class
                        self.pred_labels.append(pred_class)
                    print(f"Processed {os.path.basename(img_path)}: No ground truth, {len(pred_classes)} false positives")
                
                # If we have ground truth but no predictions
                elif true_classes and not pred_classes:
                    # All ground truths are missed (false negatives)
                    for true_class in true_classes:
                        self.true_labels.append(true_class)
                        self.pred_labels.append(self.background_idx)  # Background class
                    print(f"Processed {os.path.basename(img_path)}: {len(true_classes)} ground truths, no predictions")
                
                # If we have both ground truth and predictions
                elif true_classes and pred_classes:
                    # Match predictions to ground truth based on IOU or class
                    matched_gt = set()
                    matched_pred = set()
                    
                    # For each ground truth, find the best matching prediction
                    for i, true_class in enumerate(true_classes):
                        best_match_idx = -1
                        best_match_score = 0
                        
                        # Find predictions of the same class
                        for j, pred_class in enumerate(pred_classes):
                            if j in matched_pred:
                                continue  # Skip already matched predictions
                            
                            if pred_class == true_class:
                                # If masks are available, calculate IOU
                                if i < len(true_masks) and j < len(pred_masks) and pred_masks[j] is not None:
                                    # Use the IOU calculation from the ImageProcessor
                                    iou = self.img_processor.calculate_iou(true_masks[i], pred_masks[j])
                                    if iou > best_match_score:
                                        best_match_score = iou
                                        best_match_idx = j
                                else:
                                    # Without masks, just match by class
                                    best_match_idx = j
                                    best_match_score = 1.0
                                    break
                        
                        # If a good match is found
                        if best_match_idx >= 0:
                            self.true_labels.append(true_class)
                            self.pred_labels.append(pred_classes[best_match_idx])
                            matched_gt.add(i)
                            matched_pred.add(best_match_idx)
                        else:
                            # No matching prediction found - false negative
                            self.true_labels.append(true_class)
                            self.pred_labels.append(self.background_idx)  # Background class
                    
                    # Add remaining predictions as false positives
                    for j, pred_class in enumerate(pred_classes):
                        if j not in matched_pred:
                            self.true_labels.append(self.background_idx)  # Background class
                            self.pred_labels.append(pred_class)
                    
                    print(f"Processed {os.path.basename(img_path)}: {len(true_classes)} ground truths, {len(pred_classes)} predictions")
                    print(f"  Matched: {len(matched_gt)}, Missed: {len(true_classes) - len(matched_gt)}, False positive: {len(pred_classes) - len(matched_pred)}")
                
                # If neither ground truth nor predictions
                else:
                    print(f"Processed {os.path.basename(img_path)}: No ground truth, no predictions")
                
                self.processed_images += 1
                
            except Exception as e:
                print(f"Error processing {os.path.basename(img_path)}: {e}")
                self.skipped_images += 1
        
        print(f"Dataset processing complete. Processed {self.processed_images} images, skipped {self.skipped_images}")
        print(f"Total labels: {len(self.true_labels)} true, {len(self.pred_labels)} predicted")
        assert len(self.true_labels) == len(self.pred_labels), "Mismatch in number of labels!"

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
            
        # Check if we have an odd number of coordinates (which would be invalid)
        if coords.size % 2 != 0:
            print(f"Warning: Odd number of coordinates ({coords.size}), truncating last value")
            coords = coords[:-1]  # Remove the last element to make it even
            
        # Double check we have enough points (at least 3 pairs)
        if coords.size < 6:
            print(f"Warning: Too few coordinates after fixing ({coords.size})")
            return 0
        
        try:
            # Reshape coordinates to points format [(x1,y1), (x2,y2), ...]
            points = coords.reshape(-1, 2)
            
            # Calculate area using Shoelace formula
            x = points[:, 0]
            y = points[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            return area
        except Exception as e:
            print(f"Error calculating area: {e}, coords shape: {coords.shape}, size: {coords.size}")
            return 0
        
    def build_confusion_matrix(self):
        """Build confusion matrix from collected labels."""
        if not self.true_labels or not self.pred_labels:
            print("No labels collected. Run process_dataset first.")
            return None
            
        if len(self.true_labels) != len(self.pred_labels):
            print(f"WARNING: Inconsistent number of labels: {len(self.true_labels)} true vs {len(self.pred_labels)} predicted")
            # Take the minimum length to avoid errors
            min_len = min(len(self.true_labels), len(self.pred_labels))
            self.true_labels = self.true_labels[:min_len]
            self.pred_labels = self.pred_labels[:min_len]
        
        # Analyze the distribution of labels
        total_pairs = len(self.true_labels)
        true_pos = sum(1 for t, p in zip(self.true_labels, self.pred_labels) 
                      if t == p and t != self.background_idx and p != self.background_idx)
        false_pos = sum(1 for t, p in zip(self.true_labels, self.pred_labels) 
                       if t == self.background_idx and p != self.background_idx)
        false_neg = sum(1 for t, p in zip(self.true_labels, self.pred_labels) 
                       if t != self.background_idx and p == self.background_idx)
        true_neg = sum(1 for t, p in zip(self.true_labels, self.pred_labels)
                      if t == self.background_idx and p == self.background_idx)
        
        print(f"Label distribution analysis:")
        print(f"  Total pairs: {total_pairs}")
        print(f"  True positives: {true_pos}")
        print(f"  False positives: {false_pos}")
        print(f"  False negatives: {false_neg}")
        print(f"  True negatives: {true_neg}")
        
        # Get only the classes that actually appear in the data (excluding background)
        unique_true_classes = set(t for t in self.true_labels if t != self.background_idx)
        unique_pred_classes = set(p for p in self.pred_labels if p != self.background_idx)
        active_classes = sorted(list(unique_true_classes | unique_pred_classes))
        
        # Always include background class for false positives/negatives
        class_indices = active_classes + [self.background_idx]
        
        print(f"Classes present in the data: {[self.class_names[c] if c < len(self.class_names) else 'Background' for c in class_indices]}")
        
        # Create confusion matrix with all labels including background
        cm = confusion_matrix(
            self.true_labels, 
            self.pred_labels, 
            labels=class_indices
        )
        
        # Save complete label information for metrics calculation
        self.label_pairs = list(zip(self.true_labels, self.pred_labels))
        
        return cm, class_indices
    
    def plot_confusion_matrix(self, output_dir, normalize=True):
        """Plot and save confusion matrix."""
        cm, class_indices = self.build_confusion_matrix()
        if cm is None:
            print("Cannot generate confusion matrix. No valid class pairs found.")
            self._save_summary_metrics(output_dir)
            return
        
        # Get date string for all files
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create class names for display (include background)
        display_names = []
        for i in class_indices:
            if i < len(self.class_names):
                display_names.append(self.class_names[i])
            else:
                display_names.append("Background")
        
        # Normalize if requested
        if normalize and cm.sum(axis=1).any():
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm, nan=0)
            cm_display = np.round(cm_norm, 2)
            title = "Normalized Confusion Matrix"
            fmt = ".2f"
        else:
            cm_display = cm
            title = "Confusion Matrix"
            fmt = "d"
            
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_display, annot=True, fmt=fmt, 
                    cmap="Blues", xticklabels=display_names, 
                    yticklabels=display_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Create metrics
        metrics = self._calculate_metrics(class_indices)
            
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{date_str}.png"), dpi=300)
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, f"metrics_{date_str}.csv"))
        
        # Save detailed label distribution
        self._save_detailed_results(output_dir, date_str)
        
        # Display metrics
        print("\nClassification Metrics:")
        print(metrics_df)
        
        return cm
    
    def _save_summary_metrics(self, output_dir):
        """Save summary metrics even when no confusion matrix can be generated."""
        # Get date string
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get unique classes that appear in the data
        active_classes = sorted(list(set(t for t in self.true_labels if t != self.background_idx) | 
                                   set(p for p in self.pred_labels if p != self.background_idx)))
        
        # Count occurrences by class
        class_counts = {
            'class_id': [],
            'class_name': [],
            'true_count': [],
            'pred_count': [],
            'matches': [],
            'false_negatives': [],  # missed detections
            'false_positives': [],  # false alarms
            'true_negatives': [],   # correctly rejected
        }
        
        # Count occurrences for each class that appears in the data
        for class_idx in active_classes:
            if class_idx < len(self.class_names):
                class_name = self.class_names[class_idx]
            else:
                class_name = f"Unknown-{class_idx}"
            
            # Count all occurrences
            true_count = sum(1 for t in self.true_labels if t == class_idx)
            pred_count = sum(1 for p in self.pred_labels if p == class_idx)
            
            # Count matches
            matches = sum(1 for t, p in zip(self.true_labels, self.pred_labels) 
                        if t == class_idx and p == class_idx)
            
            # Count false negatives (missed detections)
            false_negatives = sum(1 for t, p in zip(self.true_labels, self.pred_labels)
                                if t == class_idx and p != class_idx)
            
            # Count false positives (false alarms)
            false_positives = sum(1 for t, p in zip(self.true_labels, self.pred_labels)
                                if t != class_idx and p == class_idx)
            
            # Count true negatives (correctly rejected)
            true_negatives = sum(1 for t, p in zip(self.true_labels, self.pred_labels)
                               if t != class_idx and p != class_idx)
            
            # Add to summary
            class_counts['class_id'].append(class_idx)
            class_counts['class_name'].append(class_name)
            class_counts['true_count'].append(true_count)
            class_counts['pred_count'].append(pred_count)
            class_counts['matches'].append(matches)
            class_counts['false_negatives'].append(false_negatives)
            class_counts['false_positives'].append(false_positives)
            class_counts['true_negatives'].append(true_negatives)
        
        # Create DataFrame
        summary_df = pd.DataFrame(class_counts)
        
        # Add precision, recall, F1
        summary_df['precision'] = summary_df.apply(
            lambda row: row['matches'] / row['pred_count'] if row['pred_count'] > 0 else 0, axis=1)
        summary_df['recall'] = summary_df.apply(
            lambda row: row['matches'] / row['true_count'] if row['true_count'] > 0 else 0, axis=1)
        summary_df['f1_score'] = summary_df.apply(
            lambda row: 2 * (row['precision'] * row['recall']) / (row['precision'] + row['recall']) 
            if (row['precision'] + row['recall']) > 0 else 0, axis=1)
        summary_df['accuracy'] = summary_df.apply(
            lambda row: (row['matches'] + row['true_negatives']) / 
            (row['matches'] + row['false_positives'] + row['false_negatives'] + row['true_negatives']), axis=1)
        
        # Round floating point columns
        for col in ['precision', 'recall', 'f1_score', 'accuracy']:
            summary_df[col] = summary_df[col].round(3)
            
        # Add totals row
        if not summary_df.empty:
            summary_df.loc['total'] = [
                -1, 
                'TOTAL',
                summary_df['true_count'].sum(),
                summary_df['pred_count'].sum(),
                summary_df['matches'].sum(),
                summary_df['false_negatives'].sum(),
                summary_df['false_positives'].sum(),
                summary_df['true_negatives'].sum() / len(active_classes) if active_classes else 0,  # Average TN
                summary_df['matches'].sum() / summary_df['pred_count'].sum() if summary_df['pred_count'].sum() > 0 else 0,
                summary_df['matches'].sum() / summary_df['true_count'].sum() if summary_df['true_count'].sum() > 0 else 0,
                0,  # F1 will be calculated below
                0   # Accuracy will be calculated below
            ]
            
            # Calculate F1 for total
            prec = summary_df.loc['total', 'precision']
            rec = summary_df.loc['total', 'recall']
            summary_df.loc['total', 'f1_score'] = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            summary_df.loc['total', 'f1_score'] = round(summary_df.loc['total', 'f1_score'], 3)
            
            # Calculate overall accuracy
            total_correct = summary_df['matches'].sum() + summary_df['true_negatives'].sum() / len(active_classes)
            total_samples = len(self.true_labels)
            summary_df.loc['total', 'accuracy'] = round(total_correct / total_samples, 3) if total_samples > 0 else 0
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(output_dir, f"summary_metrics_{date_str}.csv"), index=False)
        
        # Also save raw labels for debugging
        labels_df = pd.DataFrame({
            'true_label': [self.class_names[t] if t != self.background_idx and t < len(self.class_names) 
                          else 'Background' if t == self.background_idx else f'Unknown-{t}' for t in self.true_labels],
            'pred_label': [self.class_names[p] if p != self.background_idx and p < len(self.class_names)
                          else 'Background' if p == self.background_idx else f'Unknown-{p}' for p in self.pred_labels],
        })
        labels_df.to_csv(os.path.join(output_dir, f"raw_labels_{date_str}.csv"), index=False)
        
        print(f"\nSummary metrics saved to {output_dir}/summary_metrics_{date_str}.csv")
        print(summary_df)
        
        return summary_df
                
    def _save_detailed_results(self, output_dir, date_str):
        """Save detailed results breakdown."""
        # First call the summary metrics function
        summary_df = self._save_summary_metrics(output_dir)
        
        # Create a detailed per-image breakdown
        detailed_results = []
        
        # Group the true and predicted labels by image
        images = []
        true_classes_per_image = []
        pred_classes_per_image = []
        
        curr_image = None
        curr_true = []
        curr_pred = []
        
        # This is a more detailed version that would require tracking image paths during processing
        # For now, we'll just save the summary metrics
        
        return summary_df
                
    def _calculate_metrics(self, class_indices):
        """Calculate precision, recall, and F1 score for all classes."""
        # Initialize metrics dictionary
        metrics = {
            "Class": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": [],
            "Accuracy": [],
            "TP": [],
            "FP": [],
            "FN": [],
            "TN": []
        }
        
        # Calculate class-level metrics - exclude background class for average calculations
        foreground_precision = []
        foreground_recall = []
        foreground_f1 = []
        foreground_accuracy = []
        
        for class_idx in class_indices:
            # Skip background for metrics calculation
            if class_idx == self.background_idx:
                continue
                
            if class_idx < len(self.class_names):
                class_name = self.class_names[class_idx]
            else:
                class_name = f"Unknown-{class_idx}"
                
            # Count true positives, false positives, false negatives, and true negatives
            TP = sum(1 for t, p in self.label_pairs if t == class_idx and p == class_idx)
            FP = sum(1 for t, p in self.label_pairs if t != class_idx and p == class_idx)
            FN = sum(1 for t, p in self.label_pairs if t == class_idx and p != class_idx)
            TN = sum(1 for t, p in self.label_pairs if t != class_idx and p != class_idx)
            
            # Calculate precision, recall, F1 score, and accuracy
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
            
            # Add to metrics
            metrics["Class"].append(class_name)
            metrics["Precision"].append(round(precision, 3))
            metrics["Recall"].append(round(recall, 3))
            metrics["F1 Score"].append(round(f1, 3))
            metrics["Accuracy"].append(round(accuracy, 3))
            metrics["TP"].append(TP)
            metrics["FP"].append(FP)
            metrics["FN"].append(FN)
            metrics["TN"].append(TN)
            
            # Store for averaging (only foreground classes)
            foreground_precision.append(precision)
            foreground_recall.append(recall)
            foreground_f1.append(f1)
            foreground_accuracy.append(accuracy)
        
        # Calculate averages if we have metrics
        if foreground_precision:
            # Add macro average (only for foreground classes)
            metrics["Class"].append("AVERAGE")
            metrics["Precision"].append(round(np.mean(foreground_precision), 3))
            metrics["Recall"].append(round(np.mean(foreground_recall), 3))
            metrics["F1 Score"].append(round(np.mean(foreground_f1), 3))
            metrics["Accuracy"].append(round(np.mean(foreground_accuracy), 3))
            metrics["TP"].append(sum(metrics["TP"]))
            metrics["FP"].append(sum(metrics["FP"]))
            metrics["FN"].append(sum(metrics["FN"]))
            metrics["TN"].append(sum(metrics["TN"]) // len(foreground_precision) if foreground_precision else 0)
            
            # Add micro average
            total_tp = sum(metrics["TP"][:-1])  # Exclude the AVERAGE row we just added
            total_fp = sum(metrics["FP"][:-1])
            total_fn = sum(metrics["FN"][:-1])
            total_tn = sum(metrics["TN"][:-1]) // len(foreground_precision) if foreground_precision else 0
            
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            micro_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0
            
            metrics["Class"].append("MICRO-AVG")
            metrics["Precision"].append(round(micro_precision, 3))
            metrics["Recall"].append(round(micro_recall, 3))
            metrics["F1 Score"].append(round(micro_f1, 3))
            metrics["Accuracy"].append(round(micro_accuracy, 3))
            metrics["TP"].append(total_tp)
            metrics["FP"].append(total_fp)
            metrics["FN"].append(total_fn)
            metrics["TN"].append(total_tn)
        
        return metrics


def main():
    # Configuration
    test_img_folder = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/test_0/labels/images'
    test_label_folder = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/test_0/labels/labels'
    model_weights = "/mnt/hpccs01/home/wardlewo/20250205_cgras_segmentation_alive_dead/train7/weights/best.pt"
    output_dir = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/confusion_matrix_results'
    max_images = 6000  # Set max number of images to process
    min_area = 0.01  # Minimum area as fraction of image size (0.1%)
    
    # We don't need to check for shapely since it should be handled by ImageProcessor
    
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
    cm_analyzer.plot_confusion_matrix(output_dir, normalize=False)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()