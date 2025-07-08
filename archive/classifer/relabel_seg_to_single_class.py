import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

def change_2024_dataset_to_2023_dataset(label_dir, output_dir, polyp_classes=[0,1]):
        # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    polypcount = 0 

    # Iterate through label files
    for label_file in tqdm(sorted(Path(label_dir).glob("*.txt")), desc="Processing labels"):
        # Read the label file
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Modify class ID in each line
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  #  class
                #if parts[0].isdigit() and int(parts[0]) in polyp_classes:
                    #Ignore Polyps
                #    polypcount += 1
                if parts[0].isdigit() and int(parts[0]) == 2:
                    parts[0] = "1"
                    updated_lines.append(" ".join(parts))
                elif parts[0].isdigit() and int(parts[0]) == 3:
                    parts[0] = "2"
                    updated_lines.append(" ".join(parts))
                else:
                    print(f"Skipping line with class ID in {polyp_classes}: {parts[0]}")
        # Write the updated lines to the output directory
        output_path = Path(output_dir) / label_file.name
        with open(output_path, "w") as f:
            f.write("\n".join(updated_lines) + "\n")
    print(f"Processing complete. Modified label files saved in: {output_dir} ")
            
def _process_single_file(args):
    """Helper function to process a single label file."""
    label_file, output_dir, classes, DeadClasses = args
    
    # Read the label file
    with open(label_file, "r") as f:
        lines = f.readlines()

    # Modify class ID in each line
    updated_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:  #  class ID and bounding box/polygon data
            if parts[0].isdigit() and int(parts[0]) in classes: #If Alive
                parts[0] = "0" 
                updated_lines.append(" ".join(parts))
            elif parts[0].isdigit() and int(parts[0]) in DeadClasses: #If Dead
                print("Found class dead " + parts[0])
                parts[0] = "1"  
                updated_lines.append(" ".join(parts))
            else: #Any that isn't coral
                print(f"Skipping line with class ID not in {classes}: {parts[0]}")
        else:
            print(f"Skipping malformed line in {label_file}: {line}")

    # Write the updated lines to the output directory
    output_path = Path(output_dir) / label_file.name
    with open(output_path, "w") as f:
        f.write("\n".join(updated_lines) + "\n")

def change_class_to_zero(label_dir, output_dir, classes, DeadClasses = [6,7], max_workers=None):
    """
    Modifies all label files to set the class ID to 0 using multithreading.

    Args:
        label_dir (str): Path to the folder containing the label files.
        output_dir (str): Directory to save modified label files.
        classes (list): List of class IDs to convert to class 0.
        DeadClasses (list): List of class IDs to convert to class 1.
        max_workers (int): Maximum number of worker threads. If None, uses default.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all label files
    label_files = sorted(Path(label_dir).glob("*.txt"))
    
    # Prepare arguments for each file
    file_args = [(label_file, output_dir, classes, DeadClasses) for label_file in label_files]
    
    # Process files using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(_process_single_file, file_args),
            total=len(file_args),
            desc="Processing labels"
        ))

    print(f"Processing complete. Modified label files saved in: {output_dir} ")


# Example usage
if __name__ == "__main__":
    label_dir = "Data/cgras/aspa/aspa_alor117_filtered_split_tiled_balanced/val_valid_0/labels"  # Path to original labels
    change_class_to_zero(label_dir, label_dir, classes=[0, 1, 2, 3, 4, 5])  
    #change_2024_dataset_to_2023_dataset(label_dir, label_dir, polyp_classes=[0,1])
