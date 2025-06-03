import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torchmetrics.classification import ConfusionMatrix
from datetime import datetime

class ConfusionMatrixAnalyzer:
    def __init__(self, detector, dataset_analyzer, class_names, class_colours, detection_type="segmentation"):
        self.detector = detector
        self.dataset_analyzer = dataset_analyzer
        self.class_names = class_names
        self.true_labels = []
        self.pred_labels = []
        self.img_processor = dataset_analyzer.img_processor if hasattr(dataset_analyzer, 'img_processor') else None
        self.processed_images = 0
        self.skipped_images = 0
        # Add background class index as last class index + 1
        self.background_idx = len(class_names)
        self.detection_type = detection_type  
        self.label_pairs = []

    def process_dataset(self, img_list, label_list, min_area=0.001):
        """
        Process all images in the test set and collect true and predicted labels.
        Include background class for false positives/negatives.
        
        Args:
            img_list: List of paths to images
            label_list: List of paths to label files
            min_area: Minimum area threshold as fraction of image (default: 0.001 or 0.1%)
        """
        count = 0
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
                                count += 1
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
                    
                    # Create matching matrix with IoU scores
                    matching_scores = []
                    
                    # For each ground truth, calculate IoU with each prediction of the same class
                    for i, true_class in enumerate(true_classes):
                        for j, pred_class in enumerate(pred_classes):
                            if pred_class != true_class:
                                continue  # Skip different classes
                                
                            iou_score = 0
                            # If masks are available, calculate IOU
                            if i < len(true_masks) and j < len(pred_masks) and pred_masks[j] is not None:
                                if self.detection_type == "segmentation":
                                    iou_score = self.img_processor.calculate_iou(true_masks[i], pred_masks[j])
                                else:
                                    # Use bounding box IoU calculation for bbox mode
                                    iou_score = self.calculate_bbox_iou(true_masks[i], pred_masks[j])
                            elif pred_class == true_class:  
                                # Without masks, just use class matching
                                iou_score = 1.0
                                
                            if iou_score > 0:  # Only store positive matches
                                matching_scores.append((i, j, iou_score))
                                
                    # Sort all matches by IoU score (descending)
                    matching_scores.sort(key=lambda x: x[2], reverse=True)
                    
                    # Greedily match ground truth to predictions - highest IoU scores first
                    for gt_idx, pred_idx, score in matching_scores:
                        if gt_idx in matched_gt or pred_idx in matched_pred:
                            continue  # Skip if either is already matched
                            
                        # Add the match
                        self.true_labels.append(true_classes[gt_idx])
                        self.pred_labels.append(pred_classes[pred_idx])
                        matched_gt.add(gt_idx)
                        matched_pred.add(pred_idx)
                        
                    # Add unmatched ground truths as false negatives
                    for i, true_class in enumerate(true_classes):
                        if i not in matched_gt:
                            self.true_labels.append(true_class)
                            self.pred_labels.append(self.background_idx)  # Background class
                    
                    # Add unmatched predictions as false positives
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
        print(f"Count of processed masks: {count}")
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
        ground_truth_data = self.img_processor.load_ground_truth_masks(label_path)
        
        for item in ground_truth_data:
            class_label = int(item[0])
            coords = item[1:]
            
            # Calculate area based on detection type
            if self.detection_type == "segmentation":
                # Calculate polygon area for segmentation
                area = self._calculate_polygon_area(coords)
            else:
                # Calculate bbox area for bounding box
                if len(coords) == 4:  # [x_center, y_center, width, height] or [x1, y1, x2, y2]
                    if coords[2] < 1.0 and coords[3] < 1.0:  # Likely normalized YOLO format
                        area = coords[2] * coords[3]  # width * height
                    else:  # Non-normalized coordinates
                        area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                else:
                    # Fall back to polygon area
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
        For bounding boxes (4 coordinates), calculate width * height directly.
        """
        # Special case for bounding boxes (4 values)
        if coords is not None and len(coords) == 4:
            # Check if it's in YOLO format [x_center, y_center, width, height]
            if coords[2] < 1.0 and coords[3] < 1.0:  # Likely normalized YOLO format
                # Area is simply width * height for normalized coordinates
                return coords[2] * coords[3]
            else:
                # For [x1, y1, x2, y2] format, area is (x2-x1) * (y2-y1)
                return (coords[2] - coords[0]) * (coords[3] - coords[1])
        
        # For polygon coordinates, use the existing method
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
            print(f"Error calculating area: {e}, coords shape: {coords.shape if hasattr(coords, 'shape') else 'unknown'}, size: {coords.size if hasattr(coords, 'size') else len(coords)}")
            return 0

    def calculate_bbox_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.
        """
        # Convert from YOLO format [x_center, y_center, width, height] to [x1, y1, x2, y2] if needed
        if len(box1) == 4:
            if box1[2] < 1.0 and box1[3] < 1.0:  # Likely normalized YOLO format
                x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
                x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
            else:  # Already in [x1, y1, x2, y2] format
                x1_1, y1_1, x2_1, y2_1 = box1
        else:
            # If polygon points, convert to bounding box
            points = np.array(box1).reshape(-1, 2)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            x1_1, y1_1, x2_1, y2_1 = x_min, y_min, x_max, y_max
        
        if len(box2) == 4:
            if box2[2] < 1.0 and box2[3] < 1.0:  # Likely normalized YOLO format
                x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
                x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
            else:  # Already in [x1, y1, x2, y2] format
                x1_2, y1_2, x2_2, y2_2 = box2
        else:
            # If polygon points, convert to bounding box
            points = np.array(box2).reshape(-1, 2)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            x1_2, y1_2, x2_2, y2_2 = x_min, y_min, x_max, y_max
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - intersection_area
        
        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0.0

    def build_confusion_matrix(self):
        """Build confusion matrix from collected labels using PyTorch."""
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
        
        # Add detailed class distribution analysis
        self._analyze_class_distribution()
        
        # Get only the classes that actually appear in the data (excluding background)
        unique_true_classes = set(t for t in self.true_labels if t != self.background_idx)
        unique_pred_classes = set(p for p in self.pred_labels if p != self.background_idx)
        active_classes = sorted(list(unique_true_classes | unique_pred_classes))
        
        # Always include background class for false positives/negatives
        class_indices = active_classes + [self.background_idx]
        
        print(f"Classes present in the data: {[self.class_names[c] if c < len(self.class_names) else 'Background' for c in class_indices]}")
        
        # Create confusion matrix with PyTorch
        # Convert lists to PyTorch tensors
        true_tensor = torch.tensor(self.true_labels)
        pred_tensor = torch.tensor(self.pred_labels)
        
        # Find the total number of classes (including background)
        num_classes = max(class_indices) + 1
        
        # Initialize the PyTorch confusion matrix
        confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Compute the confusion matrix
        cm = confmat(pred_tensor, true_tensor)
        
        # Extract the relevant part of the confusion matrix for our active classes
        # Create indices mapping for slicing
        idx_mapping = {cls: i for i, cls in enumerate(class_indices)}
        cm_subset = torch.zeros((len(class_indices), len(class_indices)), dtype=torch.int64)
        
        for i, true_cls in enumerate(class_indices):
            for j, pred_cls in enumerate(class_indices):
                if true_cls < cm.shape[0] and pred_cls < cm.shape[1]:
                    cm_subset[i, j] = cm[true_cls, pred_cls]
        
        # Save complete label information for metrics calculation
        self.label_pairs = list(zip(self.true_labels, self.pred_labels))
        
        return cm_subset, class_indices

    def _extract_true_classes_and_coords(self, label_path, min_area=0.001):
        """
        Extract class labels and coordinates from ground truth file, filtering by area.
        """
        true_classes = []
        true_coords = []
        
        # Use the ImageProcessor function to load ground truth data
        ground_truth_data = self.img_processor.load_ground_truth_masks(label_path)
        
        for item in ground_truth_data:
            class_label = int(item[0])
            coords = item[1:]
            
            # Check if these are bounding box coordinates
            if len(coords) == 4:
                # Calculate area as width * height for normalized coordinates
                if coords[2] < 1.0 and coords[3] < 1.0:  # YOLO format [x_center, y_center, width, height]
                    area = coords[2] * coords[3]  # width * height
                else:  # [x1, y1, x2, y2] format
                    area = (coords[2] - coords[0]) * (coords[3] - coords[1])
            else:
                # Calculate polygon area for non-bbox coordinates
                area = self._calculate_polygon_area(coords)
            
            # Only include if area is above threshold
            if area >= min_area:
                true_classes.append(class_label)
                true_coords.append(coords)
            else:
                print(f"Skipping small object (class {class_label}, area {area:.6f}) in {os.path.basename(label_path)}")
        
        return true_classes, true_coords

    def _analyze_class_distribution(self):
        """Provide a detailed analysis of class distribution in the dataset."""
        if not self.true_labels or not self.pred_labels:
            return
            
        # Count occurrences of each class in true and predicted labels
        true_counts = {}
        pred_counts = {}
        
        # Create counts for each class
        for i in range(len(self.class_names) + 1):  # +1 for background
            if i < len(self.class_names):
                class_name = self.class_names[i]
            else:
                class_name = "Background"
                
            true_counts[class_name] = sum(1 for t in self.true_labels if t == i)
            pred_counts[class_name] = sum(1 for p in self.pred_labels if p == i)
        
        print("\nDetailed class distribution:")
        print("  Class         | True labels | Predicted |")
        print("  ------------- | ----------- | --------- |")
        for i in range(len(self.class_names) + 1):
            if i < len(self.class_names):
                class_name = self.class_names[i]
            else:
                class_name = "Background"
                
            print(f"  {class_name:<13} | {true_counts[class_name]:<11} | {pred_counts[class_name]:<9} |")
                
        # Create a text-based mini confusion matrix for quick reference
        print("\nSimplified confusion matrix:")
        
        # Print header
        header = "True \\ Pred |"
        for i in range(len(self.class_names)):
            header += f" {self.class_names[i][:8]} |"
        header += " Bkgnd |"
        print(header)
        
        divider = "------------ |"
        for i in range(len(self.class_names) + 1):
            divider += " ------ |"
        print(divider)
        
        # Print rows
        for i in range(len(self.class_names) + 1):
            if i < len(self.class_names):
                row_name = self.class_names[i][:8]
            else:
                row_name = "Background"
                
            row = f"{row_name:<12} |"
            
            for j in range(len(self.class_names) + 1):
                count = sum(1 for t, p in zip(self.true_labels, self.pred_labels) if t == i and p == j)
                row += f" {count:6} |"
                
            print(row)
        print()

    def plot_confusion_matrix(self, output_dir, normalize="row"):
        """
        Plot and save confusion matrix using PyTorch.
        """
        cm_tensor, class_indices = self.build_confusion_matrix()
        if cm_tensor is None:
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
        cm_display = cm_tensor
        title = "Confusion Matrix"
        if normalize:
            with torch.no_grad():
                if normalize == "row" and torch.sum(cm_tensor, dim=1).any():
                    # Row normalization (normalize by true labels)
                    row_sums = torch.sum(cm_tensor, dim=1, keepdim=True)
                    # Replace zeros with ones to avoid division by zero
                    row_sums[row_sums == 0] = 1
                    cm_display = cm_tensor.float() / row_sums.float()
                    title = "Row-Normalized Confusion Matrix"
                    
                elif normalize == "column" and torch.sum(cm_tensor, dim=0).any():
                    # Column normalization (normalize by predicted labels)
                    col_sums = torch.sum(cm_tensor, dim=0, keepdim=True)
                    # Replace zeros with ones to avoid division by zero
                    col_sums[col_sums == 0] = 1
                    cm_display = cm_tensor.float() / col_sums.float()
                    title = "Column-Normalized Confusion Matrix"
                    
                elif normalize == "all" and torch.sum(cm_tensor).item() > 0:
                    # Total normalization (normalize by total count)
                    total = torch.sum(cm_tensor)
                    cm_display = cm_tensor.float() / total.float()
                    title = "Total-Normalized Confusion Matrix"
            
        # Convert to numpy for matplotlib plotting
        cm_np = cm_display.cpu().numpy()
        
        # Dynamic figure size based on number of classes to reduce whitespace
        n_classes = len(display_names)
        
        # Calculate more compact figure dimensions with better aspect ratio
        fig_width = max(5, min(n_classes * 1.0, 10))
        fig_height = max(4, min(n_classes * 0.8, 8))
        
        # Create figure with specified size
        plt.figure(figsize=(fig_width, fig_height))
        
        # Set colormap and format based on normalization
        cmap = plt.cm.Blues
        if normalize:
            plt.imshow(cm_np, interpolation='nearest', cmap=cmap, vmin=0, vmax=1, aspect='auto')
            fmt = '.2f'
        else:
            plt.imshow(cm_np, interpolation='nearest', cmap=cmap, aspect='auto')
            fmt = 'd'
            
        plt.title(title, fontsize=14, pad=10)
        
        # Add smaller colorbar with reduced size to decrease whitespace
        cbar = plt.colorbar(fraction=0.035, pad=0.03)
        cbar.ax.tick_params(labelsize=9)
        
        # Add class ticks with adjusted spacing
        tick_marks = np.arange(len(display_names))
        fontsize = max(7, min(9, 10 - n_classes // 4))  # Smaller font for more classes
        
        # Reduce space for tick labels
        plt.xticks(tick_marks, display_names, rotation=45, ha='right', fontsize=fontsize)
        plt.yticks(tick_marks, display_names, fontsize=fontsize)
        
        # Add text annotations with tighter spacing
        thresh = (cm_np.max() + cm_np.min()) / 2.0
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                plt.text(j, i, format(cm_np[i, j], fmt),
                        ha="center", va="center",
                        fontsize=fontsize,
                        color="white" if cm_np[i, j] > thresh else "black")
        
        # Use tighter layout with less padding
        plt.tight_layout(pad=0.5)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Save with normalization type in filename
        norm_str = normalize if normalize else "raw"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{norm_str}_{date_str}.png"),
                   dpi=300, bbox_inches='tight', pad_inches=0.1)
        
        # Create metrics
        metrics = self._calculate_metrics(class_indices)
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, f"metrics_{date_str}.csv"))
        
        # Save detailed label distribution
        self._save_detailed_results(output_dir, date_str)
        
        # Display metrics
        print("\nClassification Metrics:")
        print(metrics_df)
        
        plt.close()  # Close figure to free memory
        return cm_tensor
        
    def _calculate_metrics(self, class_indices):
        """Calculate precision, recall, and F1 score for all classes using PyTorch metrics."""
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
        
        # Convert label pairs to PyTorch tensors for efficient computation
        true_tensor = torch.tensor(self.true_labels)
        pred_tensor = torch.tensor(self.pred_labels)
        
        for class_idx in class_indices:
            # Skip background for metrics calculation
            if class_idx == self.background_idx:
                continue
                
            if class_idx < len(self.class_names):
                class_name = self.class_names[class_idx]
            else:
                class_name = f"Unknown-{class_idx}"
                
            # Calculate metrics using PyTorch operations
            with torch.no_grad():
                # True positives: both true and predicted are this class
                TP = torch.sum((true_tensor == class_idx) & (pred_tensor == class_idx)).item()
                
                # False positives: true is not this class but predicted is this class
                FP = torch.sum((true_tensor != class_idx) & (pred_tensor == class_idx)).item()
                
                # False negatives: true is this class but predicted is not this class
                FN = torch.sum((true_tensor == class_idx) & (pred_tensor != class_idx)).item()
                
                # True negatives: neither true nor predicted is this class
                TN = torch.sum((true_tensor != class_idx) & (pred_tensor != class_idx)).item()
                
                # Calculate metrics
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
        
    def _save_detailed_results(self, output_dir, date_str):
        """Save detailed information about each prediction/ground truth pair."""
        if not self.label_pairs:
            print("No label pairs to save.")
            return
            
        # Create DataFrame with columns for true labels, predicted labels, and result type
        detailed_results = []
        
        for true_label, pred_label in self.label_pairs:
            # Determine the result type
            if true_label == pred_label and true_label != self.background_idx:
                result_type = "True Positive"
            elif true_label != self.background_idx and pred_label == self.background_idx:
                result_type = "False Negative"
            elif true_label == self.background_idx and pred_label != self.background_idx:
                result_type = "False Positive"
            else:
                result_type = "True Negative"
                
            # Get class names
            true_name = self.class_names[true_label] if true_label < len(self.class_names) else "Background"
            pred_name = self.class_names[pred_label] if pred_label < len(self.class_names) else "Background"
            
            detailed_results.append({
                "True_Label_Index": true_label,
                "True_Label_Name": true_name,
                "Predicted_Label_Index": pred_label,
                "Predicted_Label_Name": pred_name,
                "Result_Type": result_type
            })
            
        # Convert to DataFrame and save
        results_df = pd.DataFrame(detailed_results)
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, f"detailed_results_{date_str}.csv"), index=False)
        print(f"Detailed results saved to {os.path.join(output_dir, f'detailed_results_{date_str}.csv')}")
        
    def _save_summary_metrics(self, output_dir):
        """Save summary metrics when confusion matrix generation fails."""
        # Create basic metrics for the dataset
        metrics = {
            "Total_Images_Processed": self.processed_images,
            "Skipped_Images": self.skipped_images,
            "Total_Labels": len(self.true_labels),
            "Date_Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create DataFrame and save
        metrics_df = pd.DataFrame([metrics])
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_df.to_csv(os.path.join(output_dir, f"summary_metrics_{date_str}.csv"), index=False)
        print(f"Summary metrics saved to {os.path.join(output_dir, f'summary_metrics_{date_str}.csv')}")
