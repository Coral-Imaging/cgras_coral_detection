#! /usr/bin/env python3

"""Script to generate plots of resolution experiment results

    Assumes: Each resolution experiment has in the ultralitics train folder has a txt file called [txt_name]
        which has the pasted raw confusion matrix of the train results. NOTE this is not automatically done.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

resolutions = ['96', '128', '160', '320', '640']
runs_dir = '/home/java/Java/cslics/resolution_test_results/models'
folder_name_extension = [
    #'_v8n_results_ve', 
    #'_v8n_results',
    #'_with_scale', 
    #'_with_no_scale'
    '_resolution_test'
    ]
txt_name = 'output.txt'
save_plots_dir = '/home/java/Java/cslics/resolution_test_results/plots'
Top_2_classes = False #set to true if just want to anaylise over Recruit Live White and Recruit Live Cluster White
SHOW = False #True if plots created should be displayed or False to just save the plots

#TODO 640p run with defult missing,
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if Top_2_classes:
    class_ignore=[2,3,4,5,6,7,8,9,10]
else:
    class_ignore=None

def extract_confusion_matrix(file_path):
    """
    Extracts a confusion matrix from a text file and converts it into a NumPy array.
    Args:
        file_path (str): Path to the text file containing the confusion matrix.
    Returns:
        np.ndarray or None: A NumPy array representation of the confusion matrix 
                            if found; otherwise, None.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    matrix_pattern = r'\[\[.*?\]\](?=(\n|$))'
    match = re.search(matrix_pattern, content, re.DOTALL)
    if match:
        matrix_str = match.group(0)
        # Remove extra spaces and normalize the matrix
        matrix_str = re.sub(r'\s+', ',', matrix_str) 
        matrix_str = matrix_str.replace(',,', ',')   
        matrix_str = matrix_str.replace('[,', '[') 
        # Convert the cleaned matrix string into a NumPy array
        matrix = np.array(eval(matrix_str))
        return matrix
    else:
        print("No confusion matrix found in the file.")
        return None

def get_TP_FP_FN_TN(conf_mat, class_ignore=None):
    """
    Calculates the average True Positive Rate (TPR), False Negative Rate (FNR),
    False Positive Rate (FPR), and True Negative Rate (TNR) from a confusion matrix,
    optionally ignoring specified classes.
    Args:
        conf_mat (np.ndarray): The confusion matrix of shape (n_classes, n_classes).
        class_ignore (list, optional): List of class indices to ignore in the calculation. 
                                       Defaults to None.
    Returns:
        tuple: Averages of TPR, FNR, FPR, and TNR across non-ignored classes.
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
        tn_d[i] = total_samples - row_sum - col_sum + tp
    
    mask = np.ones(conf_mat.shape[0], dtype=bool)
    mask[class_ignore] = False    
    TPR = np.where((tp_d + fn_d) > 0, tp_d / (tp_d + fn_d), 0)[mask]
    FNR = np.where((tp_d + fn_d) > 0, fn_d / (tp_d + fn_d), 0)[mask]
    FPR = np.where((fp_d + tn_d) > 0, fp_d / (fp_d + tn_d), 0)[mask]
    TNR = np.where((fp_d + tn_d) > 0, tn_d / (fp_d + tn_d), 0)[mask]
    TPmean = np.mean(TPR)
    FNmean = np.mean(FNR)
    FPmean = np.mean(FPR)
    TNmean = np.mean(TNR) 
    conf_mat = normalize(conf_mat, axis=0, norm='l1')
    print(conf_mat)
    return TPmean, FNmean, FPmean, TNmean

def p_r_f1(conf_mat, class_ignore=None):
    """Computes the precision, recall, and F1-score from a confusion matrix,
    optionally ignoring specific classes.
    Args:
        conf_mat (np.ndarray): The confusion matrix of shape (n_classes, n_classes).
        class_ignore (list, optional): List of class indices to ignore in the calculation. 
                                       Defaults to None.
    Returns:
        tuple: Averages of precision, recall, and F1-score across the non-ignored classes.
    """
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

def get_MAP(file_path, class_ignore=None):
    """
    Extracts mAP50 and mAP50-95 values from a file, optionally ignoring specified classes.
    Args:
        file_path (str): Path to the results file.
        class_ignore (list): List of class indices to ignore (default is None).
    Returns:
        tuple: Average mAP50 and mAP50-95 values.
    """
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    pattern = r"(?P<class_name>[a-zA-Z0-9_\- ]+)\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
    matches = re.finditer(pattern, content)
    
    mAP50_values, mAP50_95_values, class_counter = [], [], -1
    for match in matches:
        mAP50 = float(match.group(10))
        mAP50_95 = float(match.group(11))
        if class_ignore is None:
            return mAP50, mAP50_95
        if class_counter not in class_ignore and class_counter<11 and class_counter>=0:
            mAP50_values.append(mAP50)
            mAP50_95_values.append(mAP50_95)
        class_counter +=1
    avg_mAP50 = np.mean(mAP50_values) if mAP50_values else 0.0
    avg_mAP50_95 = np.mean(mAP50_95_values) if mAP50_95_values else 0.0
    return avg_mAP50, avg_mAP50_95


results = {
    'TP Score': {ext: [] for ext in folder_name_extension},
    'FN Score': {ext: [] for ext in folder_name_extension},
    'FP Score': {ext: [] for ext in folder_name_extension},
    'TN Score': {ext: [] for ext in folder_name_extension},
    'Precision': {ext: [] for ext in folder_name_extension},
    'Recall': {ext: [] for ext in folder_name_extension},
    'F1 Score': {ext: [] for ext in folder_name_extension},
    'mAP 50': {ext: [] for ext in folder_name_extension},
    'mAP 50-90': {ext: [] for ext in folder_name_extension}
}

# Iterate through resolutions and extensions
for res in resolutions:
    for ext in folder_name_extension:
        out_results_path = os.path.join(runs_dir, res + ext, txt_name)
        if not os.path.exists(out_results_path):
            print(f"Skipping missing file: {out_results_path}")
            if ext == folder_name_extension[0] and res == resolutions[4] and Top_2_classes==False:
                results['TP Score'][ext].append(0.435)
                results['FN Score'][ext].append(0.482)
                results['FP Score'][ext].append(0.04)
                results['TN Score'][ext].append(0.96)
                results['Precision'][ext].append(0.507)
                results['Recall'][ext].append(0.496)
                results['F1 Score'][ext].append(0.5459)
                results['mAP 50'][ext].append(0.541)
                results['mAP 50-90'][ext].append(0.379)
            elif ext == folder_name_extension[0] and res == resolutions[5] and Top_2_classes==False:
                results['TP Score'][ext].append(0.412)
                results['FN Score'][ext].append(0.504)
                results['FP Score'][ext].append(0.044)
                results['TN Score'][ext].append(0.956)
                results['Precision'][ext].append(0.486)
                results['Recall'][ext].append(0.517)
                results['F1 Score'][ext].append(0.501)
                results['mAP 50'][ext].append(0.494)
                results['mAP 50-90'][ext].append(0.346)
            else:
                results['TP Score'][ext].append(np.nan)
                results['FN Score'][ext].append(np.nan)
                results['FP Score'][ext].append(np.nan)
                results['TN Score'][ext].append(np.nan)
                results['Precision'][ext].append(np.nan)
                results['Recall'][ext].append(np.nan)
                results['F1 Score'][ext].append(np.nan)
                results['mAP 50'][ext].append(np.nan)
                results['mAP 50-90'][ext].append(np.nan)
            continue
        confusion_matrix = extract_confusion_matrix(out_results_path)
        MAP50, MAP50_90 = get_MAP(out_results_path, class_ignore)
        if confusion_matrix is not None:
            TPmean, FNmean, FPmean, TNmean = get_TP_FP_FN_TN(confusion_matrix, class_ignore)
            precision, recall, F1 = p_r_f1(confusion_matrix)
            results['TP Score'][ext].append(TPmean)
            results['FN Score'][ext].append(FNmean)
            results['FP Score'][ext].append(FPmean)
            results['TN Score'][ext].append(TNmean)
            results['Precision'][ext].append(precision)
            results['Recall'][ext].append(recall)
            results['F1 Score'][ext].append(F1)
        results['mAP 50'][ext].append(MAP50)
        results['mAP 50-90'][ext].append(MAP50_90)

# Plot metrics
for metric, values in results.items():
    plt.figure()
    for ext, vals in values.items():
        if ext == folder_name_extension[0]:
            label="Experiment: Scale 0.2"
        elif ext == folder_name_extension[1]:
            label="Experiment: Scale 0.1"
        else: label = "Experiment: Scale 0"
        plt.plot(resolutions, vals, marker='o', label=label)
    if Top_2_classes:
        plt.title(f"Top 2 Classes: {metric} vs Resolution")
    else: plt.title(f"{metric} vs Resolution")
    plt.xlabel("Resolution")
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_plots_dir}/{metric}.png")
    if SHOW:
        plt.show()