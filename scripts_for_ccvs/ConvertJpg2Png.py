#!/usr/bin/env python3

# script to convert jpg files to png files and copy original jpg files

import os
import shutil
from PIL import Image
import multiprocessing
import time


class JpgToPngConverter:
    def __init__(self, root_dir, target_dir, target_files, target_suffix='.png', 
                 processes=None, convert_to_png=False, copy_jpg=True):
        """
        Initialize the JPG to PNG converter.

        Args:
            root_dir (str): Directory containing subfolders with JPG files
            target_dir (str): Directory to save converted PNG files and original JPGs
            target_files (list): Specific files to convert
            target_suffix (str): Target file extension (default: '.png')
            processes (int): Number of processes to use for parallel conversion
            convert_to_png (bool): Whether to convert JPG files to PNG (default: True)
            copy_jpg (bool): Whether to copy original JPG files (default: True)
        """
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.target_files = target_files
        self.target_suffix = target_suffix
        self.processes = processes or multiprocessing.cpu_count()
        self.convert_to_png = convert_to_png
        self.copy_jpg = copy_jpg
        
        # Store the original MAX_IMAGE_PIXELS value
        self.original_max_pixels = Image.MAX_IMAGE_PIXELS

    def get_subfolders(self):
        """
        Get all subfolders in the root directory.

        Returns:
            list: List of subfolder paths
        """
        subfolders = []
        try:
            subfolders = [d for d in os.listdir(self.root_dir)
                          if os.path.isdir(os.path.join(self.root_dir, d))]
            print(f"Found {len(subfolders)} subfolders in {self.root_dir}")
        except Exception as e:
            print(f"Error getting subfolders: {str(e)}")

        return subfolders

    def convert_images_in_subfolder(self, subfolder_name):
        """
        Process images in a specific subfolder based on configured options.

        Args:
            subfolder_name (str): Name of the subfolder to process

        Returns:
            tuple: (png_success_count, jpg_copy_count, files_found_count)
        """
        png_success_count = 0
        jpg_copy_count = 0
        files_found_count = 0
        source_folder = os.path.join(self.root_dir, subfolder_name)
        target_folder = os.path.join(self.target_dir, subfolder_name)

        # Safety check: Ensure source and target folders are different
        if os.path.abspath(source_folder) == os.path.abspath(target_folder):
            print(f"WARNING: Source and target folders are the same. Skipping to prevent overwriting: {source_folder}")
            return (0, 0, 0)

        # Create target subfolder if actually copying/converting files
        if self.copy_jpg or self.convert_to_png:
            os.makedirs(target_folder, exist_ok=True)

        for target_file in self.target_files:
            source_file = os.path.join(source_folder, target_file)
            if not os.path.exists(source_file):
                continue
                
            # Count found files even if not copying/converting
            files_found_count += 1
            
            # Special case: if both copy_jpg and convert_to_png are False, just print the file
            if not self.copy_jpg and not self.convert_to_png:
                print(f"Found file (would process): {source_file}")
                continue

            # Copy original JPG file if enabled
            if self.copy_jpg:
                jpg_target_path = os.path.join(target_folder, target_file)
                try:
                    shutil.copy2(source_file, jpg_target_path)
                    jpg_copy_count += 1
                    print(f"Copied original JPG: {source_file} to {jpg_target_path}")
                except Exception as e:
                    print(f"Error copying JPG {source_file}: {str(e)}")

            # Convert to PNG if enabled
            if self.convert_to_png:
                # Create the target filename with PNG extension
                target_file_name = os.path.splitext(target_file)[0] + self.target_suffix
                target_file_path = os.path.join(target_folder, target_file_name)

                try:
                    # Handle large images safely
                    # Temporarily increase the maximum image size limit
                    Image.MAX_IMAGE_PIXELS = None  # Disable the DecompressionBomb protection

                    # Open and convert image using context manager for better resource handling
                    with Image.open(source_file) as img:
                        # Use optimized settings for faster conversion
                        img.save(target_file_path, format="PNG", optimize=True, compress_level=1)
                    
                    png_success_count += 1
                    print(f"Converted {source_file} to {target_file_path}")

                    # Reset the limit to default after conversion
                    Image.MAX_IMAGE_PIXELS = self.original_max_pixels

                except Exception as e:
                    print(f"Error converting {source_file}: {str(e)}")

        return (png_success_count, jpg_copy_count, files_found_count)

    def process_all_subfolders(self):
        """
        Process all subfolders based on configured options.

        Returns:
            tuple: (processed_count, png_success_count, jpg_copy_count, files_found_count)
        """
        subfolders = self.get_subfolders()
        processed_count = 0
        total_png_success = 0
        total_jpg_copies = 0
        total_files_found = 0

        # Create the target directory if we're actually copying/converting files
        if self.copy_jpg or self.convert_to_png:
            os.makedirs(self.target_dir, exist_ok=True)

        # Special case: if both copy_jpg and convert_to_png are False, print in dry run mode
        if not self.copy_jpg and not self.convert_to_png:
            print("\nDRY RUN MODE: No files will be copied or converted.")
            print("The following files would be processed:")

        # Decide whether to use multiprocessing based on number of subfolders
        if len(subfolders) > 4 and self.processes > 1 and (self.copy_jpg or self.convert_to_png):
            print(f"Starting parallel processing with {self.processes} processes")
            start_time = time.time()
            
            with multiprocessing.Pool(processes=self.processes) as pool:
                results = pool.map(self.convert_images_in_subfolder, subfolders)
            
            # Process results from parallel execution
            for png_count, jpg_count, files_found in results:
                total_png_success += png_count
                total_jpg_copies += jpg_count
                total_files_found += files_found
            
            processed_count = len(subfolders)
            print(f"Parallel processing completed in {time.time() - start_time:.2f} seconds")
        else:
            # Process sequentially for small number of folders or if multiprocessing disabled
            start_time = time.time()
            for subfolder in subfolders:
                print(f"\nProcessing subfolder: {subfolder}")
                png_count, jpg_count, files_found = self.convert_images_in_subfolder(subfolder)
                processed_count += 1
                total_png_success += png_count
                total_jpg_copies += jpg_count
                total_files_found += files_found
            
            if self.copy_jpg or self.convert_to_png:
                print(f"Sequential processing completed in {time.time() - start_time:.2f} seconds")

        return (processed_count, total_png_success, total_jpg_copies, total_files_found)

    def run(self):
        """
        Main method to execute the JPG processing workflow.
        """
        print(f"Starting JPG processing")
        print(f"Source directory: {self.root_dir}")
        print(f"Target directory: {self.target_dir}")
        print(f"Target files: {self.target_files}")
        print(f"Converting to PNG: {self.convert_to_png}")
        print(f"Copying JPG: {self.copy_jpg}")

        start_time = time.time()
        processed, png_success, jpg_copied, files_found = self.process_all_subfolders()
        total_time = time.time() - start_time

        print(f"\nProcessing summary:")
        print(f"Processed {processed} subfolders")
        print(f"Found {files_found} target files")
        
        if self.convert_to_png:
            print(f"Successfully converted {png_success} files to PNG")
        if self.copy_jpg:
            print(f"Successfully copied {jpg_copied} original JPG files")
            
        # Only show execution time if we actually did work
        if self.copy_jpg or self.convert_to_png:
            print(f"Total execution time: {total_time:.2f} seconds")
            
        print("Processing completed!")


def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Copy and convert image files')
    parser.add_argument('--root-dir', type=str, 
                        default="/home/dtsai/cgras_data/detector/data/2024Oct",
                        help='Root directory containing subfolders with JPG files')
    parser.add_argument('--target-dir', type=str, 
                        default="/home/dtsai/Data/cgras_datasets/cgras_2024_amag_manual_comparison",
                        help='Directory to save output files')
    parser.add_argument('--processes', type=int,
                        default=4,
                        help='Number of parallel processes to use')
    parser.add_argument('--no-png', action='store_false',
                        help='Skip PNG conversion, only copy JPG files')
    parser.add_argument('--no-jpg', action='store_true',
                        help='Skip JPG copying, only convert to PNG')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show files that would be processed, without copying or converting')
    
    args = parser.parse_args()

    target_files = [
        'rotated_whole_reco_original_image_annotated.jpg',
        'rotated_whole_reco_original_image_grid.jpg',
        'rotated_whole_reco_image_annotated.jpg'
    ]

    target_suffix = '.png'

    # Determine conversion options
    convert_to_png = not args.no_png
    copy_jpg = not args.no_jpg
    
    # Override both flags if dry-run is set
    if args.dry_run:
        convert_to_png = False
        copy_jpg = False

    # Create and run the converter
    converter = JpgToPngConverter(
        root_dir=args.root_dir, 
        target_dir=args.target_dir, 
        target_files=target_files, 
        target_suffix=target_suffix, 
        processes=args.processes,
        convert_to_png=convert_to_png,
        copy_jpg=copy_jpg
    )
    converter.run()


if __name__ == "__main__":
    main()