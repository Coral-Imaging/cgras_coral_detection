import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path = "/home/java/Java/hpc-home/20250205_cgras_segmentation_alive_dead/train7/results.csv"
df = pd.read_csv(file_path)

# Display basic information and the first few rows
df.info(), df.head()


# Set up the figure
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Plot Loss Curves
axes[0].plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", linestyle="--", color="blue")
axes[0].plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", linestyle="-", color="blue")
axes[0].plot(df["epoch"], df["train/seg_loss"], label="Train Seg Loss", linestyle="--", color="green")
axes[0].plot(df["epoch"], df["val/seg_loss"], label="Val Seg Loss", linestyle="-", color="green")
axes[0].plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss", linestyle="--", color="red")
axes[0].plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss", linestyle="-", color="red")
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid()

# Plot Performance Metrics
axes[1].plot(df["epoch"], df["metrics/precision(B)"], label="Precision (B)", linestyle="-", color="blue")
axes[1].plot(df["epoch"], df["metrics/recall(B)"], label="Recall (B)", linestyle="-", color="green")
axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50 (B)", linestyle="-", color="red")
axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95 (B)", linestyle="-", color="purple")
axes[1].plot(df["epoch"], df["metrics/precision(M)"], label="Precision (M)", linestyle="--", color="blue")
axes[1].plot(df["epoch"], df["metrics/recall(M)"], label="Recall (M)", linestyle="--", color="green")
axes[1].plot(df["epoch"], df["metrics/mAP50(M)"], label="mAP50 (M)", linestyle="--", color="red")
axes[1].plot(df["epoch"], df["metrics/mAP50-95(M)"], label="mAP50-95 (M)", linestyle="--", color="purple")
axes[1].set_title("Performance Metrics")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Metric Value")
axes[1].legend()
axes[1].grid()

# Show plots
plt.tight_layout()
plt.show()
