import os
import shutil
import glob
import cv2 as cv
from datetime import datetime
from ultralytics import YOLO
from view_predictions import save_image_predictions_mask
from Utils import classes, class_colours
import torch
import numpy as np


class Prediction:
    def __init__(self, img_name, predicted_data, confidences, class_labels, results):
        self.img_name = img_name
        self.predicted_data = predicted_data
        self.confidences = confidences
        self.class_labels = class_labels
        self.results = results


class Detector:
    def __init__(self, model_weights, device=None):
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        
        self.model = YOLO(model_weights).to(self.device)
    
    def predict(self, img_path):
        img_name = os.path.basename(img_path)
        results = self.model.predict(img_path, save=True, imgsz=640, agnostic_nms=True, iou=0.5, task='segment')
        
        for r in results:
            tensor = r.boxes
            confidences = [b.conf.item() for b in tensor]
            class_labels = [b.cls.item() for b in tensor]
            return Prediction(img_name, tensor, confidences, class_labels, r)
        
        return None


class ImageProcessor:
    def __init__(self, classes, class_colours):
        self.classes = classes
        self.class_colours = class_colours

    def visualize_predictions(self, prediction_list, output_dir, img_folder, label_folder):
        for prediction in prediction_list:
            img_name = prediction.img_name
            img_path = os.path.join(img_folder, img_name)
            label_path = os.path.join(label_folder, img_name.replace('.jpg', '.txt'))
            image = cv.imread(img_path)
            conf = prediction.confidences
            class_list = prediction.class_labels
            results = prediction.results

            save_image_predictions_mask(
                results, image, img_path, os.path.join(output_dir, "visualise"), 
                conf, class_list, self.classes, self.class_colours, ground_truth=True, 
                txt=label_path, negDataMine=True
            )
    
    @staticmethod
    def calculate_iou(mask1_coords, mask2_coords, img_shape=(640, 640)):
        """Calculate Intersection over Union (IoU) between two sets of polygon coordinates."""
        # Create empty binary masks
        mask1 = np.zeros(img_shape, dtype=np.uint8)
        mask2 = np.zeros(img_shape, dtype=np.uint8)

        # Convert coordinates to integer pixel values
        mask1_coords = np.array(mask1_coords, dtype=np.int32).reshape((-1, 1, 2))
        mask2_coords = np.array(mask2_coords, dtype=np.int32).reshape((-1, 1, 2))

        # Fill the binary masks with the polygon
        cv.fillPoly(mask1, [mask1_coords], 1)
        cv.fillPoly(mask2, [mask2_coords], 1)

        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        return intersection / union if union > 0 else 0

    @staticmethod
    def load_ground_truth_masks(label_path):
        """
        Load ground truth masks from a label file.
        
        Args:
            label_path: Path to the label file
            
        Returns:
            List of ground truth masks, where each mask is [class_label, x1, y1, x2, y2, ...]
        """
        ground_truth_masks = []
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
                if lines:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) > 1:  # Ensure the line has the expected number of parts
                            class_label = float(parts[0])
                            coords = list(map(float, parts[1:]))
                            ground_truth_masks.append([class_label] + coords)
                        else:
                            print(f"Warning: Skipping malformed line in {label_path}: {line.strip()}")
                else:
                    print(f"Warning: The label file {label_path} is empty.")
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
        
        return ground_truth_masks


class DatasetAnalyzer:
    def __init__(self, img_processor, iou_threshold=0.5):
        self.img_processor = img_processor
        self.iou_threshold = iou_threshold


    def compare_predictions_with_labels(self, predicted_list, label_list):
        """Compare model predictions with ground truth labels to identify False Positives (FP) and False Negatives (FN)."""
        fp_list, fn_list = [], []

        for prediction, label_path in zip(predicted_list, label_list):
            img_name = prediction.img_name
            results = prediction.results

            # Load predicted masks
            predicted_masks = []
            predicted_class_labels = []
            for i in range(len(results)):
                mask_data = results.masks.xyn[i]
                class_labels = results.boxes.cls[i]
                predicted_class_labels.append(class_labels)
                predicted_masks.append(mask_data)
                
            # Load ground truth masks using the new function
            ground_truth_masks = self.img_processor.load_ground_truth_masks(label_path)

            # When both predicted and ground truth masks are empty, skip the image
            if not predicted_masks and not ground_truth_masks:
                continue        
                
            # Identify False Negatives (FN): ground truth objects with no matching prediction
            if len(ground_truth_masks) > len(predicted_masks):
                print(f"More ground truth masks than predicted masks for {img_name}")
                fn_list.append(prediction)
            else:
                found_fn = False
                for gt_mask in ground_truth_masks:
                    matched = False
                    for pred_mask in predicted_masks:
                        gt_points = gt_mask[1:]
                        pred_points = pred_mask[1:]
                        try:
                            if (self.img_processor.calculate_iou(gt_points, pred_points) >= self.iou_threshold):
                                matched = True
                                break
                        except Exception as e:
                            print(f"Error calculating IoU for {img_name}: {e}")
                    if not matched:
                        found_fn = True
                        break
                        
                if found_fn:
                    fn_list.append(prediction)

            # Identify False Positives (FP): predicted objects with no matching ground truth
            if len(predicted_masks) > len(ground_truth_masks):
                print(f"More predicted masks than ground truth masks for {img_name}")
                fp_list.append(prediction)
            else:
                found_fp = False
                for pred_mask in predicted_masks:
                    matched = False
                    for gt_mask in ground_truth_masks:
                        gt_points = gt_mask[1:]
                        pred_points = pred_mask[1:]
                        try:
                            if (self.img_processor.calculate_iou(gt_points, pred_points) >= self.iou_threshold):
                                matched = True
                                break
                        except Exception as e:
                            print(f"Error calculating IoU for {img_name}: {e}")
                    if not matched:
                        found_fp = True
                        break
                        
                if found_fp:
                    fp_list.append(prediction)

        return fp_list, fn_list


class FileManager:
    @staticmethod
    def create_output_dirs(base_dir, date_str):
        fp_output_dir = os.path.join(base_dir, f"FP_test_{date_str}")
        fn_output_dir = os.path.join(base_dir, f"FN_test_{date_str}")

        for output_dir in [fp_output_dir, fn_output_dir]:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "visualise"), exist_ok=True)

        return fp_output_dir, fn_output_dir

    @staticmethod
    def move_files(file_list, src_dir, dest_dir):
        for file in file_list:
            shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))

    @staticmethod
    def move_fp_fn_files(fp_list, fn_list, num_fp, num_fn, test_img_folder, test_label_folder, fp_output_dir, fn_output_dir):
        # Extract filenames from Prediction objects
        fp_filenames = [pred.img_name for pred in fp_list[:num_fp]]
        fn_filenames = [pred.img_name for pred in fn_list[:num_fn]]
        
        # Move false positive images and labels
        FileManager.move_files(fp_filenames, test_img_folder, os.path.join(fp_output_dir, "images"))
        FileManager.move_files([f.replace('.jpg', '.txt') for f in fp_filenames], test_label_folder, os.path.join(fp_output_dir, "labels"))

        # Move false negative images and labels
        FileManager.move_files(fn_filenames, test_img_folder, os.path.join(fn_output_dir, "images"))
        FileManager.move_files([f.replace('.jpg', '.txt') for f in fn_filenames], test_label_folder, os.path.join(fn_output_dir, "labels"))


def main():
    # Configuration
    test_img_folder = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/test_0/labels/images'
    test_label_folder = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/test_0/labels/labels'
    model_weights = '/mnt/hpccs01/home/wardlewo/ultralytics_output/runs/20250326_cgras_segmentation_2022-2023_dataset_alive_dead/20250326_8n_train_multiGpu_B512_best/weights/best.pt'
    output_base_dir = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/test_FN_FP'
    percentage = 0.15  # Editable percentage of images to move
    
    print("Starting False Positive and False Negative Analysis")
    
    # Initialize classes
    detector = Detector(model_weights)
    img_processor = ImageProcessor(classes, class_colours)
    dataset_analyzer = DatasetAnalyzer(img_processor)
    file_manager = FileManager()
    
    # Get list of test images and labels
    img_list = sorted(glob.glob(os.path.join(test_img_folder, '*.jpg')))
    label_list = sorted(glob.glob(os.path.join(test_label_folder, '*.txt')))

    # Set maximum number of images to process for testing
    max_images = min(6000, len(img_list))  # Change this value as needed for testing
    
    # Run model on test set and collect predictions
    predicted_list = []
    for img_path in img_list[:max_images]:
        prediction = detector.predict(img_path)
        if prediction:
            print(f"Predicted Array for {prediction.img_name}: {prediction.class_labels}")
            predicted_list.append(prediction)

    print("Predicted List Length: ", len(predicted_list))

    # Compare predictions with ground truth labels to identify FP and FN
    fp_list, fn_list = dataset_analyzer.compare_predictions_with_labels(
        predicted_list, label_list[:max_images]
    )

    # Calculate number of images to move
    num_fp = int(len(fp_list) * percentage)
    num_fn = int(len(fn_list) * percentage)

    # Create output directories
    date_str = datetime.now().strftime("%Y%m%d")
    fp_output_dir, fn_output_dir = file_manager.create_output_dirs(output_base_dir, date_str)

    # Move false positive and false negative images and labels (commented out for safety)
    # file_manager.move_fp_fn_files(
    #     fp_list, fn_list, num_fp, num_fn, test_img_folder, test_label_folder, 
    #     fp_output_dir, fn_output_dir
    # )

    # Generate visualizations 
    img_processor.visualize_predictions(fp_list, fp_output_dir, test_img_folder, test_label_folder)
    img_processor.visualize_predictions(fn_list, fn_output_dir, test_img_folder, test_label_folder)


if __name__ == "__main__":
    main()