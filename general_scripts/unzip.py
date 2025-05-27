import zipfile
import os

def unzip_file(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped {zip_path} to {extract_to}")

if __name__ == "__main__":
    zip_path = '/mnt/hpccs01/home/wardlewo/Data/cgras/job_2427841_dataset_2025_05_20_05_29_33_ultralytics yolo segmentation 1.0.zip'  # Replace with your .zip file path
    extract_to = '/mnt/hpccs01/home/wardlewo/Data/cgras/2024_cgras_pdae/20250521_Pdae_100'  # Replace with your extraction directory path
    unzip_file(zip_path, extract_to)