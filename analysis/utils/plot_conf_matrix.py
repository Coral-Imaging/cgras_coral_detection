from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(self, output_dir, normalize="row"):
    """
    Plot and save confusion matrix using PyTorch.
    
    Args:
        output_dir: Directory to save output files
        normalize: Normalization method - one of ["row", "column", "all", None]
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