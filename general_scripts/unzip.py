import zipfile
import os

def unzip_file(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped {zip_path} to {extract_to}")

if __name__ == "__main__":
    zip_path = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_split+patches/cgras_labels_fixed_split_n_tilled.zip'  # Replace with your .zip file path
    extract_to = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_split+patches'  # Replace with your extraction directory path
    unzip_file(zip_path, extract_to)