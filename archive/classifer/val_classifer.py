#! /usr/bin/env python3

"""val_segmenter.py
validate a model against a dataset. Gets TP FP FN TN, precision and recall scores.
"""
#TODO: no of images in datset, info on what dataset is.
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weights_file = '/home/java/Java/Cgras/classifier_model/train_all_coral/weights/best.pt' #model

# Load a model
model = YOLO(weights_file)

# Validate the model
#if not arguments, will use the defult arguments and validate on the Val files of the model trained dataset.
metrics_d = model.val( classes = [0,1,2,3,4,5,6,7],
                    batch = 1, #Cls head needs batch 1 or doesn't work with current yolov8 version
                    project = "/home/java/Java/ultralytics_output/train_all_coral", 
                    data='/home/java/Java/hpc-home/Data/cgras/classifier/classifer_split', 
                    plots=True)
 
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


def plot_results(data):
    """plot_results
    Plot the data in a heatmap
    """
    x_labels = ['P (Positive)', 'N (Negative)']  # x-axis labels
    y_labels = ['T (True)', 'F (False)']  # y-axis labels
    plt.figure(figsize=(6, 4))
    sns.heatmap(data, annot=True, fmt=".2%", cmap='Blues', xticklabels=x_labels, yticklabels=y_labels)
    plt.title(f"Simplified confusion matrix")
    plt.show()

TPmean, FNmean, FPmean, TNmean = get_TP_FP_FN_TN(conf_mat_d, class_ignore=[0,1,7])
data = np.array([[TPmean, TNmean], [FPmean, FNmean]])
plot_results(data)

precision, recall, F1 = p_r_f1(conf_mat_d, class_ignore=[6, 7, 8, 9, 10])

print("---------------Results---------------------")

print(f"TP={TPmean:.4f}, FP={FPmean:.4f}, FN={FNmean:.4f}, TN={TNmean:.4f}")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={F1:.4f}")

print("Done")
import code
code.interact(local=dict(globals(), **locals()))

print(metrics_d.top1) #print top1 accuracy
print(metrics_d.top5) #print top5 accuracy
metrics_d.confusion_matrix.print() #print RAW confusion metrix
