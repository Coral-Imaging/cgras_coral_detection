import os
from pathlib import Path
from tqdm import tqdm




def change_class_to_zero(label_dir, output_dir, classes, DeadClasses = [6,7]):
    """
    Modifies all label files to set the class ID to 0.

    Args:
        label_dir (str): Path to the folder containing the label files.
        output_dir (str): Directory to save modified label files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through label files
    for label_file in tqdm(sorted(Path(label_dir).glob("*.txt")), desc="Processing labels"):
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

    print(f"Processing complete. Modified label files saved in: {output_dir} ")


# Example usage
if __name__ == "__main__":
    label_dir = "/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+RemovedNegs/train_4/labels/labels"  # Path to original labels
    output_dir = "/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/train_4/labels/labels"  # Path to save modified labels
    change_class_to_zero(label_dir, output_dir, classes=[0, 1, 2, 3, 4, 5])  
