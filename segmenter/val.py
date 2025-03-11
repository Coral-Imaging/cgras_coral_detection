#! /usr/bin/env python3

""" val_segmenter.py

    Validate a YOLO model against a dataset. Computes True Positives (TP), False Positives (FP),
    False Negatives (FN), True Negatives (TN), precision, recall, and F1 scores.
"""

from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIGURATION: MODIFY AS NEEDED
# -----------------------------
PROJECT_DIR = "cgras_segmentation"
WEIGHTS_FILE = "cgras_segmentation/train11/weights/best.pt"
DATA_FILE = "/media/agoni/RRAP03/outputs/training_not_contained/cgras_data.yaml"
CONF_THRESH = 0.25
IOU_THRESH = 0.6
IGNORE_CLASSES = None

# -----------------------------
# MODEL INITIALIZATION & VALIDATION
# -----------------------------
model = YOLO(WEIGHTS_FILE)
metrics_d = model.val(conf=CONF_THRESH, iou=IOU_THRESH, project=PROJECT_DIR, data=DATA_FILE, plots=True)

conf_mat_d = metrics_d.confusion_matrix.matrix  # Confusion matrix
conf_mat_normalized = conf_mat_d / (conf_mat_d.sum(0).reshape(1, -1) + 1E-9)  # Normalize

# -----------------------------
# FUNCTION DEFINITIONS
# -----------------------------
def compute_confusion_metrics(conf_mat, ignore_classes=None):
    """Compute TP, FP, FN, TN, ignoring specified classes."""
    if ignore_classes is None:
        ignore_classes = []
    
    tp = conf_mat.diagonal()
    fp = conf_mat.sum(1) - tp
    fn = conf_mat.sum(0) - tp
    total_samples = conf_mat.sum()
    tn = total_samples - (fp + fn + tp)
    
    mask = np.ones(conf_mat.shape[0], dtype=bool)
    mask[ignore_classes] = False
    
    tp_rate = np.where(tp + fn > 0, tp / (tp + fn), 0)[mask]
    fn_rate = np.where(tp + fn > 0, fn / (tp + fn), 0)[mask]
    fp_rate = np.where(fp + tn > 0, fp / (fp + tn), 0)[mask]
    tn_rate = np.where(fp + tn > 0, tn / (fp + tn), 0)[mask]
    
    return np.mean(tp_rate), np.mean(fn_rate), np.mean(fp_rate), np.mean(tn_rate)

def compute_precision_recall_f1(conf_mat, ignore_classes=None):
    """Compute precision, recall, and F1-score, ignoring specified classes."""
    if ignore_classes is None:
        ignore_classes = []
    
    mask = np.ones(conf_mat.shape[0], dtype=bool)
    mask[ignore_classes] = False
    
    tp = conf_mat.diagonal()
    fp = conf_mat.sum(1) - tp
    fn = conf_mat.sum(0) - tp
    
    precision = np.where(tp + fp > 0, tp / (tp + fp), 0)[mask]
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0)[mask]
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    
    return np.mean(precision), np.mean(recall), np.mean(f1)

def plot_confusion_matrix(data, conf_thresh, iou_thresh):
    """Plot a simplified confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(data, annot=True, fmt=".2%", cmap='Blues', xticklabels=['P (Positive)', 'N (Negative)'], yticklabels=['T (True)', 'F (False)'])
    plt.title(f"Confusion Matrix (Conf={conf_thresh:.2f}, IOU={iou_thresh:.2f})")
    plt.show()

# -----------------------------
# COMPUTE & DISPLAY RESULTS
# -----------------------------
TPmean, FNmean, FPmean, TNmean = compute_confusion_metrics(conf_mat_d, ignore_classes=IGNORE_CLASSES)
data_matrix = np.array([[TPmean, TNmean], [FPmean, FNmean]])
plot_confusion_matrix(data_matrix, CONF_THRESH, IOU_THRESH)

precision, recall, f1_score = compute_precision_recall_f1(conf_mat_d, ignore_classes=IGNORE_CLASSES)

print("--------------- Results ---------------------")
print(f"TP={TPmean:.4f}, FP={FPmean:.4f}, FN={FNmean:.4f}, TN={TNmean:.4f}")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
print(f"mAP50-95={metrics_d.maps}")


# Visualization for segmentation model
print(f"MAP50-90 for all classes: {metrics_d.seg.map}")
print(f"MAP50 for all classes: {metrics_d.seg.map50}")
print(f"MAP50-90 per class: {metrics_d.seg.maps}")
print(f"MAP50 per class: {metrics_d.seg.ap50}")

print("Done")
