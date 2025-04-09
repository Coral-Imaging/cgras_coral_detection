#! /usr/bin/env python3

"""val_segmenter.py
validate a model against a dataset. Gets TP FP FN TN, precision and recall scores.
"""
#TODO: no of images in datset, info on what dataset is.
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weights_file = '/mnt/hpccs01/home/wardlewo/2024_seg8n.pt' #model
PROJECT_DIR = "runs/val/20250326_cgras_segmentation_DS2023_8n"
CONF_THRESH = 0.5
IOU_THRESH = 0.5
IGNORE_CLASSES = None
data_file = '/mnt/hpccs01/home/wardlewo/Corals/cgras_settler_counter/segmenter/combined_2023_2024.yaml'
# -----------------------------
# MODEL INITIALIZATION & VALIDATION
# -----------------------------
model = YOLO(weights_file)
metrics_d = model.val(conf=CONF_THRESH, iou=IOU_THRESH, project=PROJECT_DIR, data=data_file, plots=True)

# Get confusion matrix
conf_mat_d = metrics_d.confusion_matrix.matrix  # Confusion matrix

# Access the segmentation mAP@50
map50 = metrics_d.seg.map50

# Access precision, recall and F1
precision = metrics_d.seg.p  # Precision
recall = metrics_d.seg.r     # Recall
f1 = metrics_d.seg.f1        # F1 score

# Print the values
print(f"Segmentation mAP@50: {map50:.4f}")

# Handle precision, recall, and F1 which may be arrays
if hasattr(precision, 'shape') and precision.size > 1:
    # If it's an array with multiple values (per class)
    print(f"Precision (mean): {np.mean(precision):.4f}")
    print("Precision per class:")
    class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(len(precision))]
    for i, p in enumerate(precision):
        print(f"  {class_names[i]}: {p:.4f}")
else:
    # If it's a single value
    print(f"Precision: {float(precision):.4f}")

# Do the same for recall
if hasattr(recall, 'shape') and recall.size > 1:
    print(f"Recall (mean): {np.mean(recall):.4f}")
    print("Recall per class:")
    class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(len(recall))]
    for i, r in enumerate(recall):
        print(f"  {class_names[i]}: {r:.4f}")
else:
    print(f"Recall: {float(recall):.4f}")

# And for F1
if hasattr(f1, 'shape') and f1.size > 1:
    print(f"F1 Score (mean): {np.mean(f1):.4f}")
    print("F1 Score per class:")
    class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(len(f1))]
    for i, f in enumerate(f1):
        print(f"  {class_names[i]}: {f:.4f}")
else:
    print(f"F1 Score: {float(f1):.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
conf_mat_norm = conf_mat_d.astype('float') / (conf_mat_d.sum(axis=1)[:, np.newaxis] + 1e-6)  # normalize

# Get class names if available
class_names = metrics_d.names if hasattr(metrics_d, 'names') else [f'Class {i}' for i in range(conf_mat_d.shape[0])]

# Create heatmap
sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
           xticklabels=class_names,
           yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/confusion_matrix.png")
plt.show()