import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def find_valid_folders(root_folder):
    """Find folders that contain both 'images' and 'labels' subfolders."""
    valid_folders = []
    for dirpath, dirnames, _ in os.walk(root_folder):
        if 'images' in dirnames and 'labels' in dirnames:
            valid_folders.append(dirpath)
    return valid_folders

def process_folder(labels_folder, images_folder, destination_folder, root_folder):
    """Process a single folder containing images and labels."""
    # Get lists of label files
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    # Separate non-empty and empty label files
    non_empty_labels = [f for f in label_files if os.path.getsize(os.path.join(labels_folder, f)) > 0]
    empty_labels = [f for f in label_files if os.path.getsize(os.path.join(labels_folder, f)) == 0]
    print(f"Processing {labels_folder}:")
    print("Number of non-empty labels found:", len(non_empty_labels))
    print("Number of empty labels found:", len(empty_labels))
    
    # Ensure equal counts of non-empty and empty labels
    selected_empty_labels = empty_labels[:len(non_empty_labels)]
    selected_labels = non_empty_labels + selected_empty_labels
    print("Total number of labels in new dataset:", len(selected_labels))
    
    for label_file in selected_labels:
        label_path = os.path.join(labels_folder, label_file)
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_folder, image_file)
        
        if os.path.exists(image_path):
            # Define relative path for maintaining structure
            relative_path = os.path.relpath(labels_folder, root_folder)
            label_dest = os.path.join(destination_folder, relative_path, "labels", label_file)
            image_dest = os.path.join(destination_folder, relative_path, "images", image_file)
            
            # Create subdirectories if not exist
            os.makedirs(os.path.dirname(label_dest), exist_ok=True)
            os.makedirs(os.path.dirname(image_dest), exist_ok=True)
            
            # Copy files to destination
            shutil.copy(label_path, label_dest)
            shutil.copy(image_path, image_dest)
        else:
            print(f"Image not found for label: {label_file}")

def main():
    root_folder = "/home/java/hpc-home/data/train100/split_dataset/exported100"
    destination_folder = "/home/java/Java/hpc-home/Data/cgras/cgras_23_n_24_combined/cgras_22_23_data_tilled_reduced_negs"
    
    valid_folders = find_valid_folders(root_folder)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for folder in valid_folders:
            labels_folder = os.path.join(folder, "labels")
            images_folder = os.path.join(folder, "images")
            futures.append(executor.submit(process_folder, labels_folder, images_folder, destination_folder, root_folder))
        
        for future in futures:
            future.result()  # Wait for all threads to complete
    
    print("File transfer complete.")

if __name__ == "__main__":
    main()