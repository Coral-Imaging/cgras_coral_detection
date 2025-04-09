import zipfile
import os
import time

def unzip_file(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.namelist())
        start_time = time.time()
        
        for i, file in enumerate(zip_ref.namelist()):
            zip_ref.extract(file, extract_to)
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (i + 1)) * total_files
            remaining_time = estimated_total_time - elapsed_time
            print(f"Extracted {i + 1}/{total_files} files. Estimated time remaining: {remaining_time:.2f} seconds")
    
    print(f"Unzipped {zip_path} to {extract_to}")

if __name__ == "__main__":
    zip_path = '/mnt/hpccs01/home/wardlewo/Data/cgras/amil_2024/split_dataset/exported100/Amil_2024.zip'  # Replace with your .zip file path
    extract_to = '/mnt/hpccs01/home/wardlewo/Data/cgras/amil_2024/split_dataset/exported100'  # Replace with your extraction directory path
    unzip_file(zip_path, extract_to)