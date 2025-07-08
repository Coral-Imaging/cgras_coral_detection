import zipfile
import os
import time
from multiprocessing import Pool, Manager, cpu_count
import math

def extract_file_chunk(args):
    """Extract a chunk of files from the zip archive in a single process"""
    zip_path, file_chunk, extract_to, progress_dict, process_id = args
    
    extracted_count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in file_chunk:
                try:
                    zip_ref.extract(file_name, extract_to)
                    extracted_count += 1
                    
                    # Update progress every 50 files per process
                    if extracted_count % 50 == 0:
                        progress_dict[process_id] = extracted_count
                        
                except Exception as e:
                    print(f"Error extracting {file_name}: {e}")
                    
        # Final update
        progress_dict[process_id] = extracted_count
        return extracted_count
        
    except Exception as e:
        print(f"Process {process_id} error: {e}")
        return extracted_count

def unzip_file(zip_path, extract_to, max_workers=None):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Determine optimal number of workers for HPC
    if max_workers is None:
        max_workers = cpu_count() 
    
    print(f"Using {max_workers} processes for extraction...")
    
    # Get file list
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)
    
    print(f"Starting extraction of {total_files} files...")
    
    # Split files into chunks for each process
    chunk_size = math.ceil(total_files / max_workers)
    file_chunks = [file_list[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    # Create shared progress tracking
    manager = Manager()
    progress_dict = manager.dict()
    
    # Prepare arguments for each process
    process_args = []
    for i, chunk in enumerate(file_chunks):
        if chunk:  # Only add non-empty chunks
            progress_dict[i] = 0
            process_args.append((zip_path, chunk, extract_to, progress_dict, i))
    
    start_time = time.time()
    
    # Start multiprocessing
    with Pool(processes=len(process_args)) as pool:
        # Start async processes
        results = pool.map_async(extract_file_chunk, process_args)
        
        # Monitor progress
        while not results.ready():
            time.sleep(2)  # Check every 2 seconds
            total_completed = sum(progress_dict.values())
            
            if total_completed > 0:
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / total_completed) * total_files
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"Extracted {total_completed}/{total_files} files. "
                      f"Estimated time remaining: {remaining_time:.2f} seconds")
        
        # Get final results
        extraction_counts = results.get()
    
    total_time = time.time() - start_time
    total_extracted = sum(extraction_counts)
    
    print(f"Unzipped {zip_path} to {extract_to}")
    print(f"Total extraction time: {total_time:.2f} seconds")
    print(f"Total files extracted: {total_extracted}/{total_files}")
    print(f"Average files per second: {total_extracted/total_time:.2f}")

if __name__ == "__main__":
    zip_path = 'Data/cslics/2024_spawn_tanks_data/100000000846a7ff.zip'  # Replace with your .zip file path
    extract_to = 'Data/cslics/2024_spawn_tanks_data/'  # Replace with your extraction directory path
    
    # For HPC, you can specify max_workers based on your system
    # e.g., unzip_file(zip_path, extract_to, max_workers=16)
    unzip_file(zip_path, extract_to)