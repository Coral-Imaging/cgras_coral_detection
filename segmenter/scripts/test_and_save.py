import os
import glob
import cv2 as cv
import yaml
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "cgras_segmentation/train11/weights/best.pt"
DATA_YAML = "/media/agoni/RRAP03/outputs/training_not_contained/cgras_data.yaml"
OUTPUTS_DIR = "/media/agoni/RRAP03/outputs/test_and_save/"
CONF_THRESH = 0.25
IOU_THRESH = 0.6
IGNORE_CLASSES = None

# Load dataset information from YAML file
with open(DATA_YAML, "r") as file:
    dataset_info = yaml.safe_load(file)

CLASS_NAMES = dataset_info.get("names", {})
TEST_IMAGES_DIRS = [os.path.join(dataset_info["path"], path) for path in dataset_info["test"]]
TEST_LABELS_DIRS = [os.path.join(dataset_info["path"], path.replace("images", "labels")) for path in dataset_info["test"]]

# Generate unique test output directory
def get_new_test_dir(base_path):
    test_dirs = sorted(glob.glob(os.path.join(base_path, "test_*")))
    test_number = len(test_dirs) + 1
    new_test_dir = os.path.join(base_path, f"test_{test_number}")
    os.makedirs(new_test_dir, exist_ok=True)
    return new_test_dir

test_output_dir = get_new_test_dir(OUTPUTS_DIR)

# Create subfolders for classification outcomes
outcome_types = ["TP", "FP", "FN", "TN"]
outcome_dirs = {t: os.path.join(test_output_dir, t) for t in outcome_types}
for d in outcome_dirs.values():
    os.makedirs(d, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Define colors for different classes
CLASS_COLORS = {
    int(cls_id): tuple(np.random.randint(0, 255, 3).tolist()) for cls_id in CLASS_NAMES
}

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Run inference
for test_dir, label_dir in zip(TEST_IMAGES_DIRS, TEST_LABELS_DIRS):
    image_files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        results = model(img_path, conf=CONF_THRESH, save=False)
        detections = results[0].boxes
        
        # Load original image
        img = cv.imread(img_path)
        img_gt = img.copy()
        img_pred = img.copy()
        
        # Load ground truth labels
        gt_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = list(map(float, line.strip().split()))
                    if len(values) >= 3:
                        cls_id = int(values[0])
                        polygon_points = np.array(values[1:]).reshape(-1, 2)
                        polygon_points[:, 0] *= img.shape[1]
                        polygon_points[:, 1] *= img.shape[0]
                        x1, y1 = np.min(polygon_points, axis=0)
                        x2, y2 = np.max(polygon_points, axis=0)
                        gt_labels.append((cls_id, [int(x1), int(y1), int(x2), int(y2)]))
                        cv.rectangle(img_gt, (int(x1), int(y1)), (int(x2), int(y2)), CLASS_COLORS.get(cls_id, (255, 255, 255)), 2)
        
        # Predicted bounding boxes
        pred_boxes = detections.xyxy.cpu().numpy() if len(detections) > 0 else np.array([])
        pred_labels = detections.cls.cpu().numpy() if len(detections) > 0 else np.array([])
        
        for i, pred_box in enumerate(pred_boxes):
            x1, y1, x2, y2 = map(int, pred_box)
            color = CLASS_COLORS.get(int(pred_labels[i]), (255, 255, 255))
            cv.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
        
        # Create subplot for visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv.cvtColor(img_gt, cv.COLOR_BGR2RGB))
        ax[0].set_title("Ground Truth")
        ax[0].axis("off")
        
        ax[1].imshow(cv.cvtColor(img_pred, cv.COLOR_BGR2RGB))
        ax[1].set_title("Predicted")
        ax[1].axis("off")
        
        # Create legend
        legend_patches = [plt.Line2D([0], [0], color=np.array(CLASS_COLORS[cls_id])/255.0, lw=4, label=CLASS_NAMES[cls_id]) for cls_id in CLASS_NAMES]
        fig.legend(handles=legend_patches, loc='upper center', ncol=4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(test_output_dir, f"comparison_{img_name}.png"))
        plt.close()

print(f"Test results saved to: {test_output_dir}")