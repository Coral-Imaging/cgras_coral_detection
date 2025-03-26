# Dataset Processing Workflow

This repository contains scripts to process CVAT annotated datasets through a multi-stage workflow:

1. **Balance Dataset**: Remove excess empty images to balance positive and negative examples
2. **Split Dataset**: Divide into train/validation/test sets
3. **Tile Images**: Create overlapping tiles within each split

## Overview

The new workflow improves on the previous approach by:

1. First balancing the dataset to ensure equal distribution of empty and non-empty labels
2. Then splitting into train/val/test to maintain proper distribution across splits
3. Finally tiling within each split to avoid data leakage between splits

## Installation

No additional installation is required beyond the existing dependencies:
- Python 3.6+
- OpenCV (cv2)
- NumPy
- PIL (Pillow)
- matplotlib
- shapely
- PyYAML

## Usage

### Basic Usage

The simplest way to run the workflow is with the `run_workflow.py` script:

```bash
python run_workflow.py --data_path "/media/java/RRAP03" --input "export100_from_cvat" --output "processed_dataset"
```

### Command Line Options

```
--data_path      Base path for all operations
--input         Input folder name (relative to data_path)
--output        Output folder name (will be created under data_path/outputs)
--tile_size     Tile size in pixels, default: 640x640
--overlap       Overlap percentage between tiles, default: 50
--train_split   Training split ratio, default: 0.7
--val_split     Validation split ratio, default: 0.15
--test_split    Test split ratio, default: 0.15
--skip_balance  Skip the balance step if specified
--verbose       Print detailed progress information
```

### Directory Structure

The script expects the following input structure:
```
DATA_PATH/
└── INPUT_FOLDER/
    ├── dataset1/
    │   ├── images/
    │   │   └── *.jpg
    │   ├── labels/
    │   │   └── *.txt
    │   ├── data.yaml
    │   └── Train.txt
    ├── dataset2/
    │   └── ...
    └── ...
```

And creates the following output structure:
```
DATA_PATH/
└── outputs/
    └── OUTPUT_FOLDER/
        ├── balanced_dataset/
        │   └── ...
        ├── split_dataset/
        │   └── ...
        ├── final_dataset/
        │   ├── dataset1/
        │   │   ├── train/
        │   │   │   ├── images/
        │   │   │   └── labels/
        │   │   ├── valid/
        │   │   │   ├── images/
        │   │   │   └── labels/
        │   │   └── test/
        │   │       ├── images/
        │   │       └── labels/
        │   └── ...
        └── cgras_data.yaml
```

## Scripts

### Main Scripts

- `process_dataset.py`: Main script to run the entire workflow
- `run_workflow.py`: User-friendly command-line interface to the workflow

### Utility Classes

- `neg_remover.py`: Balances the dataset by ensuring equal numbers of empty and non-empty labels
- `file_splitter.py`: Splits the dataset into train/validation/test sets
- `image_tiler.py`: Tiles images and labels with configurable overlap

## Example

```bash
# Process a dataset with default parameters
python run_workflow.py --data_path "/media/java/RRAP03" --input "export100_from_cvat" --output "processed_dataset"

# Process with overlap
python run_workflow.py --data_path "/media/java/RRAP03" --input "export100_from_cvat" --output "processed_dataset" --overlap 30

# Skip the balancing step
python run_workflow.py --data_path "/media/java/RRAP03" --input "export100_from_cvat" --output "processed_dataset" --skip_balance
```

## Output

The final output is a `cgras_data.yaml` file that contains paths to all the processed datasets:

```yaml
names:
  0: alive
  1: dead
  2: mask_live
  3: mask_dead
path: /media/java/RRAP03/outputs/processed_dataset/final_dataset
train:
- dataset1/train/images
- dataset1/train_0/images
- dataset1/train_1/images
val:
- dataset1/valid/images
test:
- dataset1/test/images
```

This file can be used directly for training models.