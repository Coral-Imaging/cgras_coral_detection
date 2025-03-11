#! /usr/bin/env python3

"""val_segmenter.py
validate a model against a dataset. Gets TP FP FN TN, precision and recall scores.
"""

#TODO: no of images in datset, info on what dataset is.
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weights_file = 'cgras_segmentation/train3/weights/best.pt'
conf_thresh = 0.25
iou_thresh = 0.6

# Load a model
model = YOLO(weights_file)

# Validate the model
#if not arguments, will use the defult arguments and validate on the Val files of the model trained dataset.
metrics_d = model.val(conf=conf_thresh, iou=iou_thresh, project = "cgras_segmentation", data='/media/agoni/RRAP03/outputs/cgras_data.yaml', plots=True) #data='/media/wardlewo/cslics_ssd/SCU_Pdae_Data/split and tilling/cgras_20230421.yaml',

tp_d, fp_d = metrics_d.confusion_matrix.tp_fp() # returns 2 arrays, 1xN where N is the number of classes.
conf_mat_d = metrics_d.confusion_matrix.matrix #has the confusion matrix as NXN array
conf_mat_normalised = conf_mat_d / (conf_mat_d.sum(0).reshape(1, -1) + 1E-9)

def get_TP_FP_FN_TN(conf_mat, class_ignore):
    """get_TP_FP_FN_TN
        Get the average True Positive, False Positive, False Negative and True Negative 
        rates for the confusion matrix ignoring the classes in class_ignore.
    """
    if class_ignore is None:
        class_ignore = []
    
    tp_d = conf_mat.diagonal()
    fp_d = conf_mat.sum(1) - tp_d
    fn_d = conf_mat.sum(0) - tp_d
    total_samples = conf_mat.sum()
    tn_d = np.zeros(conf_mat.shape[0])
    for i in range(conf_mat.shape[0]):
        tp = conf_mat[i, i]
        row_sum = conf_mat[i, :].sum()
        col_sum = conf_mat[:, i].sum()
    tp_d = conf_mat.diagonal()
    fp_d = conf_mat.sum(1) - tp_d
    fn_d = conf_mat.sum(0) - tp_d
    total_samples = conf_mat.sum()
    tn_d = np.zeros(conf_mat.shape[0])
    for i in range(conf_mat.shape[0]):
        tp = conf_mat[i, i]
        row_sum = conf_mat[i, :].sum()
        col_sum = conf_mat[:, i].sum()
        tn_d[i] = total_samples - row_sum - col_sum + tp

    mask = np.ones(conf_mat.shape[0], dtype=bool)
    mask[class_ignore] = False

    TPR = np.where((tp_d+ fn_d) > 0, tp_d/ (tp_d+ fn_d), 0)[mask]
    FNR = np.where((tp_d+ fn_d) > 0, fn_d / (tp_d+ fn_d), 0)[mask]
    FPR = np.where((fp_d+ tn_d) > 0, fp_d/ (fp_d+ tn_d), 0)[mask]
    TNR = np.where((fp_d+ tn_d) > 0, tn_d / (fp_d+ tn_d), 0)[mask]
    TPmean = np.mean(TPR)
    FNmean = np.mean(FNR)
    FPmean = np.mean(FPR)
    TNmean = np.mean(TNR)
    return TPmean, FNmean, FPmean, TNmean


def p_r_f1(conf_mat, class_ignore=None):
    if class_ignore is None:
        class_ignore = []
    
    mask = np.ones(conf_mat.shape[0], dtype=bool)
    mask[class_ignore] = False

    tp_d = conf_mat.diagonal()
    fp_d = conf_mat.sum(1) - tp_d
    fn_d = conf_mat.sum(0) - tp_d

    precision = np.where((tp_d + fp_d) > 0, tp_d / (tp_d + fp_d), 0)[mask]
    recall = np.where((tp_d + fn_d) > 0, tp_d / (tp_d + fn_d), 0)[mask]
    F1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)

    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_F1 = np.mean(F1)

    return avg_precision, avg_recall, avg_F1


def plot_results(data, conf_thresh, iou_thresh):
    """plot_results
    Plot the data in a heatmap
    """
    x_labels = ['P (Positive)', 'N (Negative)']  # x-axis labels
    y_labels = ['T (True)', 'F (False)']  # y-axis labels
    plt.figure(figsize=(6, 4))
    sns.heatmap(data, annot=True, fmt=".2%", cmap='Blues', xticklabels=x_labels, yticklabels=y_labels)
    plt.title(f"Simplified confusion matrix with confidence of {conf_thresh:.2f} and IOU of {iou_thresh:.2f}")
    plt.show()

TPmean, FNmean, FPmean, TNmean = get_TP_FP_FN_TN(conf_mat_d, class_ignore=None)
data = np.array([[TPmean, TNmean], [FPmean, FNmean]])
plot_results(data, conf_thresh, iou_thresh)

precision, recall, F1 = p_r_f1(conf_mat_d, class_ignore=None)

print("---------------Results---------------------")

print(f"TP={TPmean:.4f}, FP={FPmean:.4f}, FN={FNmean:.4f}, TN={TNmean:.4f}")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={F1:.4f}")
print(f"mAP50={metrics_d.map50} mAP50-95={metrics_d.map:.4f}")
## Visulise instances

print(f"MAP50-90 for all classes{metrics_d.seg.map}")
print(f"MAP50 for all classes {metrics_d.seg.map50}")
print(f"MAP50-90 for each class: {metrics_d.seg.maps}") #including classes not trained on
print(f"MAP50 for each class: {metrics_d.seg.ap50}") #excluding classes not trained on

print("Done")
import code
code.interact(local=dict(globals(), **locals()))


metrics_d.box.map    # map50-95
metrics_d.box.map50  # map50
metrics_d.box.map75  # map75
metrics_d.box.maps   # a list contains map50-95 of each category
metrics_d.box.p   # Precision for each class
metrics_d.box.r   # Recall for each class
metrics_d.box.all_ap  # AP scores for all classes and all IoU thresholds
metrics_d.box.ap_class_index  # Index of class for each AP
metrics_d.box.nc  # Number of classes
metrics_d.box.ap50  # AP at IoU threshold of 0.5 for all classes
metrics_d.box.ap  # AP at IoU thresholds from 0.5 to 0.95 for all classes
metrics_d.box.mp  # Mean precision of all classes
metrics_d.box.mr  # Mean recall of all classes
metrics_d.box.map50  # Mean AP at IoU threshold of 0.5 for all classes
metrics_d.box.map75  # Mean AP at IoU threshold of 0.75 for all classes
metrics_d.box.map  # Mean AP at IoU thresholds from 0.5 to 0.95 for all classes
metrics_d.box.mean_results  # Mean of results
metrics_d.box.class_result  # Class-aware result
metrics_d.box.maps  # mAP of each class
metrics_d.box.fitness  # Model fitness as a weighted combination of metrics_d
metrics_d.confusion_matrix.print() #print RAW confusion metrix
