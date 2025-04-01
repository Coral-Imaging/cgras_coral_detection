#! /usr/bin/env python3

""" tiling_images.py
script created to tile big images into smaller ones with annotations that can then be trained on via a yolo model
"""

import os
import numpy as np
import cv2 as cv
import glob
from PIL import Image
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
from shapely.validation import explain_validity
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse
import signal
import sys

full_res_dir = '/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/Reduced_dataset_patches/fixxed_labels/valid'
save_path = '/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/Reduced_dataset_patches'

#full_res_dir = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/dec_17_split/train/'
#save_path = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/dec_17_split_n_tilled/'

# Add argument parser for command line options
parser = argparse.ArgumentParser(description='Tile images and annotations for YOLO training')
parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
parser.add_argument('--tile_width', type=int, default=640, help='Tile width (default: 640)')
parser.add_argument('--tile_height', type=int, default=640, help='Tile height (default: 640)')
parser.add_argument('--truncate', type=float, default=0.5, help='Minimum percentage of object that must be in tile (default: 0.5)')
parser.add_argument('--max_files', type=int, default=16382, help='Maximum files per directory (default: 16382)')
args = parser.parse_args()

# Use command line arguments if provided
TILE_WIDTH = args.tile_width
TILE_HEIGHT = args.tile_height
TRUNCATE_PERCENT = args.truncate
max_files = args.max_files

# Set number of workers
num_workers = args.workers if args.workers is not None else multiprocessing.cpu_count()
print(f"Using {num_workers} worker processes")

# Extract the prefix from the last part of full_res_dir
prefix = os.path.basename(full_res_dir)

#images in one folder, labels in another. Only want to do images with an ossociated label file
imglist = sorted(glob.glob(os.path.join(full_res_dir, 'images', '*.jpg')))
TILE_OVERLAP = round((TILE_HEIGHT+TILE_WIDTH)/2 * TRUNCATE_PERCENT)
directory_count = 0
file_counter = 0   

def make_sub_dirctory_save(prefix, save_path):
    save_train = os.path.join(save_path, f'{prefix}_{directory_count}')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_train, exist_ok=True)
    save_images = os.path.join(save_train, 'images')
    save_labels = os.path.join(save_train, 'labels')
    os.makedirs(save_images, exist_ok=True)
    os.makedirs(save_labels, exist_ok=True)
    return save_images, save_labels

save_img, save_labels = make_sub_dirctory_save(prefix, save_path)

def is_mostly_contained(polygon, x_start, x_end, y_start, y_end, threshold):
    """Returns true if a Shaply polygon has more then threshold percent in the area of a specified bounding box."""
    polygon_box = box(*polygon.bounds)
    tile_box = box(x_start, y_start, x_end, y_end)
    if not polygon.is_valid:
        explanation = explain_validity(polygon)
        #print(f"Invalid Polygon: {explanation} at {x_start}_{y_start}")
        return False
    if not polygon_box.intersects(tile_box):
        return False
    intersection = polygon.intersection(tile_box)
    return intersection.area > (threshold * polygon.area)

def truncate_polygon(polygon, x_start, x_end, y_start, y_end):
    """Returns a polygon with points constrained to a specified bounding box."""
    tile_box = box(x_start, y_start, x_end, y_end)
    intersection = polygon.intersection(tile_box)
    return intersection

def create_polygon_unnormalised(parts, img_width, img_height):
    """Creates a Polygon from unnormalized part coordinates, as [class_ix, xn, yn ...]"""
    xy_coords = [round(float(p) * img_width) if i % 2 else round(float(p) * img_height) for i, p in enumerate(parts[1:], start=1)]
    polygon_coords = [(xy_coords[i], xy_coords[i + 1]) for i in range(0, len(xy_coords), 2)]
    polygon = Polygon(polygon_coords)
    return polygon

def normalise_polygon(truncated_polygon, class_number, x_start, x_end, y_start, y_end, width, height):
    """Normalize coordinates of a polygon with respect to a specified bounding box."""
    points = []
    if isinstance(truncated_polygon, Polygon):
        x_coords, y_coords = truncated_polygon.exterior.coords.xy
        xy = [class_number]

        for c, d in zip(x_coords, y_coords):
            x_val = 1.0 if c == x_end else (c - x_start) / width
            y_val = 1.0 if d == y_end else (d - y_start) / height
            xy.extend([x_val, y_val])

        points.append(xy)
        
    elif isinstance(truncated_polygon, (MultiPolygon, GeometryCollection)):
        for p in truncated_polygon.geoms:
            points.append(normalise_polygon(p, class_number, x_start, x_end, y_start, y_end, width, height))
    return points

def cut_n_save_img(x_start, x_end, y_start, y_end, np_img, img_save_path):
    """Save a tile section of an image given by a bounding box"""
    cut_tile = np.zeros(shape=(TILE_WIDTH, TILE_HEIGHT, 3), dtype=np.uint8)
    cut_tile[0:TILE_HEIGHT, 0:TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
    cut_tile_img = Image.fromarray(cut_tile)
    cut_tile_img.save(img_save_path)

def cut_annotation(x_start, x_end, y_start, y_end, lines, imgw, imgh):
    """From instance lines in label file, find objects in the bounding box and return the renormalised xy points if there are any"""
    writeline = []
    incomplete_lines = set()  
    for j, line in enumerate(lines):
        parts = line.split()
        class_number = int(parts[0])
        polygon = create_polygon_unnormalised(parts, imgw, imgh)

        if len(parts) < 1 or polygon.is_empty:
            if j not in incomplete_lines: 
                print(f"line {j} is incomplete")
                incomplete_lines.add(j)
                import code
                code.interact(local=dict(globals(), **locals()))    
            continue
        if is_mostly_contained(polygon, x_start, x_end, y_start, y_end, TRUNCATE_PERCENT):
            truncated_polygon = truncate_polygon(polygon, x_start, x_end, y_start, y_end)
            xyn = normalise_polygon(truncated_polygon, class_number, x_start, x_end, y_start, y_end, TILE_WIDTH, TILE_HEIGHT)
            writeline.append(xyn)
    return writeline


#cut the image
def cut(img_name, save_img, test_name, save_labels, txt_name, img_no):
    """Cut a image into tiles, save the annotations renormalised"""
    pil_img = Image.open(img_name, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)
    img = cv.imread(img_name)
    imgw, imgh = img.shape[1], img.shape[0]
    x_tiles = (imgw + TILE_WIDTH - TILE_OVERLAP - 1) // (TILE_WIDTH - TILE_OVERLAP)
    y_tiles = (imgh + TILE_HEIGHT - TILE_OVERLAP - 1) // (TILE_HEIGHT - TILE_OVERLAP)
    total_tiles = x_tiles * y_tiles  # Total tiles to process
    with tqdm(total=total_tiles, desc=f"Processing image {os.path.basename(img_name)[:-4]}", unit="tile") as pbar:
        for x in range(x_tiles):
            for y in range(y_tiles):
                x_end = min((x + 1) * TILE_WIDTH - TILE_OVERLAP * (x != 0), imgw)
                x_start = x_end - TILE_WIDTH
                y_end = min((y + 1) * TILE_HEIGHT - TILE_OVERLAP * (y != 0), imgh)
                y_start = y_end - TILE_HEIGHT

                img_save_path = os.path.join(save_img, f"{test_name}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.jpg")
                txt_save_path = os.path.join(save_labels, f"{test_name}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.txt")
                #make cut and save image
                cut_n_save_img(x_start, x_end, y_start, y_end, np_img, img_save_path)
                #cut annotaion and save
                with open(txt_name, 'r') as file:
                    lines = file.readlines()
                try:
                    writeline = cut_annotation(x_start, x_end, y_start, y_end, lines, imgw, imgh)
                except:
                    print("error in cut_annotations")
                    import code
                    code.interact(local=dict(globals(), **locals()))    

                with open(txt_save_path, 'w') as file:
                    for line in writeline:
                        file.write(" ".join(map(str, line)).replace('[', '').replace(']', '').replace(',', '') + "\n")
                pbar.update(1)
                # import code
                # code.interact(local=dict(globals(), **locals()))        

def calculate_img_section_no(img_name):
    """return the number of tiles made from one image depending on tile width and img size"""
    pil_img = Image.open(img_name, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)
    img = cv.imread(img_name)
    imgw, imgh = img.shape[1], img.shape[0]
    # Count number of sections to make
    x_tiles = (imgw + TILE_WIDTH - TILE_OVERLAP - 1) // (TILE_WIDTH - TILE_OVERLAP)
    y_tiles = (imgh + TILE_HEIGHT - TILE_OVERLAP - 1) // (TILE_HEIGHT - TILE_OVERLAP)
    return x_tiles*y_tiles

# Modified to return information about the tile instead of writing directly
def process_tile(args):
    """Process a single tile from an image"""
    img_name, txt_name, test_name, x_start, x_end, y_start, y_end, imgw, imgh = args
    
    # Read image once outside of the loop to avoid repeated disk I/O
    pil_img = Image.open(img_name, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)
    
    # Cut the tile from the image
    cut_tile = np.zeros(shape=(TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    cut_tile[0:TILE_HEIGHT, 0:TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
    
    # Process annotations
    with open(txt_name, 'r') as file:
        lines = file.readlines()
    try:
        writeline = cut_annotation(x_start, x_end, y_start, y_end, lines, imgw, imgh)
    except Exception as e:
        return None, None, str(e)
    
    return (test_name, x_start, y_start, cut_tile, writeline)

def cut_parallel(img_name, save_img, test_name, save_labels, txt_name, img_no):
    """Cut an image into tiles in parallel, save the annotations renormalized"""
    # Read image dimensions once
    img = cv.imread(img_name)
    imgw, imgh = img.shape[1], img.shape[0]
    
    # Calculate step size for 50% overlap
    step_size = TILE_WIDTH // 2
    
    # Calculate number of tiles in each dimension
    x_tiles = (imgw - TILE_WIDTH) // step_size + 1
    y_tiles = (imgh - TILE_HEIGHT) // step_size + 1
    
    # Add extra tile if needed to cover the image
    if x_tiles * step_size + TILE_WIDTH < imgw:
        x_tiles += 1
    if y_tiles * step_size + TILE_HEIGHT < imgh:
        y_tiles += 1
        
    total_tiles = x_tiles * y_tiles
    
    # Prepare arguments for each tile
    tile_args = []
    for x in range(x_tiles):
        x_start = x * step_size
        x_end = min(x_start + TILE_WIDTH, imgw)
        
        # Adjust x_start if we're at the edge
        if x_end == imgw:
            x_start = max(0, imgw - TILE_WIDTH)
            
        for y in range(y_tiles):
            y_start = y * step_size
            y_end = min(y_start + TILE_HEIGHT, imgh)
            
            # Adjust y_start if we're at the edge
            if y_end == imgh:
                y_start = max(0, imgh - TILE_HEIGHT)
                
            tile_args.append((img_name, txt_name, test_name, x_start, x_end, y_start, y_end, imgw, imgh))
    
    # Process tiles in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_tile, args) for args in tile_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing image {test_name}", unit="tile"):
            results.append(future.result())
    
    # Save results to disk
    for result in results:
        if result[0] is None:  # Error occurred
            continue
        
        test_name, x_start, y_start, cut_tile, writeline = result
        
        # Save image
        img_save_path = os.path.join(save_img, f"{test_name}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.jpg")
        cut_tile_img = Image.fromarray(cut_tile)
        cut_tile_img.save(img_save_path)
        
        # Save annotation
        txt_save_path = os.path.join(save_labels, f"{test_name}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.txt")
        with open(txt_save_path, 'w') as file:
            for line in writeline:
                file.write(" ".join(map(str, line)).replace('[', '').replace(']', '').replace(',', '') + "\n")
    
    return total_tiles

# Global flag for handling KeyboardInterrupt
stop_processing = False

def signal_handler(sig, frame):
    """Handle interrupt signals to cleanly stop processing"""
    global stop_processing
    print("\nCaught signal, cleaning up...")
    stop_processing = True
    print("Will exit after current image completes...")
    # Don't exit immediately to allow cleanup

# Register the signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_image(img_path, txt_path, save_img_dir, save_labels_dir, test_name):
    """Process a complete image and its annotations, cutting into tiles"""
    try:
        # Read image
        img = cv.imread(img_path)
        if img is None:
            return 0, f"Failed to read image {img_path}"

        imgw, imgh = img.shape[1], img.shape[0]
        
        # Read image with PIL for better handling
        try:
            pil_img = Image.open(img_path)
            np_img = np.array(pil_img)
        except Exception as e:
            return 0, f"Error reading image with PIL: {e}"
        
        # Read label file
        try:
            with open(txt_path, 'r') as file:
                lines = file.readlines()
        except Exception as e:
            return 0, f"Failed to read label file {txt_path}: {e}"
        
        # Calculate tile grid parameters
        step_size_x = TILE_WIDTH - TILE_OVERLAP
        step_size_y = TILE_HEIGHT - TILE_OVERLAP
        
        x_tiles = (imgw - TILE_WIDTH + step_size_x) // step_size_x
        y_tiles = (imgh - TILE_HEIGHT + step_size_y) // step_size_y
        
        # Handle edge cases
        if x_tiles * step_size_x + TILE_WIDTH < imgw:
            x_tiles += 1
        if y_tiles * step_size_y + TILE_HEIGHT < imgh:
            y_tiles += 1
        
        tiles_created = 0
        
        # Process each tile sequentially
        for x in range(x_tiles):
            x_start = x * step_size_x
            x_end = min(x_start + TILE_WIDTH, imgw)
            
            # Adjust x_start if we're at the edge
            if x_end == imgw:
                x_start = max(0, imgw - TILE_WIDTH)
                
            for y in range(y_tiles):
                y_start = y * step_size_y
                y_end = min(y_start + TILE_HEIGHT, imgh)
                
                # Adjust y_start if we're at the edge
                if y_end == imgh:
                    y_start = max(0, imgh - TILE_HEIGHT)
                
                # Cut the tile from the image
                cut_tile = np.zeros(shape=(TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
                cut_tile[0:TILE_HEIGHT, 0:TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
                
                # Generate filenames
                img_save_path = os.path.join(save_img_dir, f"{test_name}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.jpg")
                txt_save_path = os.path.join(save_labels_dir, f"{test_name}_{str(x_start).zfill(4)}_{str(y_start).zfill(4)}.txt")
                
                # Process annotations
                try:
                    writeline = cut_annotation(x_start, x_end, y_start, y_end, lines, imgw, imgh)
                except Exception as e:
                    print(f"Error processing annotations for {test_name} at {x_start},{y_start}: {e}")
                    continue
                
                # Save image
                try:
                    cut_tile_img = Image.fromarray(cut_tile)
                    cut_tile_img.save(img_save_path)
                except Exception as e:
                    print(f"Error saving tile image {img_save_path}: {e}")
                    continue
                
                # Save annotations
                try:
                    with open(txt_save_path, 'w') as file:
                        for line in writeline:
                            file.write(" ".join(map(str, line)).replace('[', '').replace(']', '').replace(',', '') + "\n")
                except Exception as e:
                    print(f"Error saving annotation file {txt_save_path}: {e}")
                    continue
                
                tiles_created += 1
                
        return tiles_created, "Success"
    
    except Exception as e:
        return 0, f"Error processing image: {str(e)}"

def process_image_list(img_list, start_idx=0, max_images=None):
    """Process a list of images, with support for resuming from a specific index"""
    global stop_processing
    
    start_time = time.time()
    processed_count = 0
    total_tiles = 0
    
    try:
        # Process images in a loop, optionally limited by max_images
        for i, img in enumerate(img_list[start_idx:]):
            idx = i + start_idx
            
            # Stop if reached max_images
            if max_images is not None and i >= max_images:
                print(f"Reached maximum number of images ({max_images})")
                break
                
            # Check for interrupt
            if stop_processing:
                print(f"Processing stopped at image index {idx}")
                break
            
            # Get image paths
            name = os.path.basename(img)[:-4]
            img_name = os.path.join(full_res_dir, 'images', name + '.jpg')
            txt_name = os.path.join(full_res_dir, 'labels', name + '.txt')
            
            # Check if both files exist
            if not os.path.exists(txt_name):
                print(f"No text file for image {name}, skipping")
                continue
                
            # Check if output directory needs to be changed
            global file_counter, directory_count, save_img, save_labels
            files_to_add = calculate_img_section_no(img_name)
            if file_counter + files_to_add >= max_files:
                print(f'Directory full, as {file_counter} files already made and {files_to_add} will be made with next image')
                directory_count += 1
                file_counter = 0
                save_img, save_labels = make_sub_dirctory_save(prefix, save_path)
            
            # Process the image
            print(f'Processing image {idx+1}/{len(img_list)}: {name}')
            tiles_created, status = process_image(img_name, txt_name, save_img, save_labels, name)
            
            if tiles_created > 0:
                processed_count += 1
                total_tiles += tiles_created
                file_counter += tiles_created
                print(f"Image {name}: {status} - Created {tiles_created} tiles")
            else:
                print(f"Image {name}: {status}")
            
            # Save progress checkpoint
            save_checkpoint(idx + 1, processed_count, total_tiles)
            
    except Exception as e:
        print(f"Error in process_image_list: {e}")
    finally:
        # Calculate and print statistics
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nProcessed {processed_count} images ({total_tiles} tiles) in {duration:.2f} seconds")
        if processed_count > 0:
            print(f"Average: {duration/processed_count:.2f} seconds/image, {total_tiles/duration:.2f} tiles/second")
        
        return processed_count, total_tiles

def save_checkpoint(image_index, processed_count, total_tiles):
    """Save a checkpoint file to allow resuming progress"""
    checkpoint = {
        'image_index': image_index,
        'processed_count': processed_count,
        'total_tiles': total_tiles,
        'directory_count': directory_count,
        'file_counter': file_counter,
        'timestamp': time.time()
    }
    
    try:
        checkpoint_path = os.path.join(save_path, f"{prefix}_checkpoint.txt")
        with open(checkpoint_path, 'w') as f:
            for key, value in checkpoint.items():
                f.write(f"{key}={value}\n")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint():
    """Load the checkpoint file if it exists"""
    checkpoint_path = os.path.join(save_path, f"{prefix}_checkpoint.txt")
    checkpoint = {
        'image_index': 0,
        'processed_count': 0,
        'total_tiles': 0,
        'directory_count': 0,
        'file_counter': 0,
    }
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        if key in ['image_index', 'processed_count', 'total_tiles', 'directory_count', 'file_counter']:
                            checkpoint[key] = int(float(value))
            print(f"Loaded checkpoint: resuming from image {checkpoint['image_index']}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return checkpoint

# Replace the main execution part with the robust version
if __name__ == "__main__":
    # Add resume argument
    parser.add_argument('--resume', action='store_true', help='Resume from the last checkpoint')
    parser.add_argument('--start', type=int, default=0, help='Start processing from this image index')
    parser.add_argument('--count', type=int, default=None, help='Process this many images (default: all)')
    args = parser.parse_args()
    
    # Initialize or load checkpoint
    start_idx = 0
    if args.resume:
        checkpoint = load_checkpoint()
        directory_count = checkpoint['directory_count']
        file_counter = checkpoint['file_counter']
        start_idx = checkpoint['image_index']
        save_img, save_labels = make_sub_dirctory_save(prefix, save_path)
    elif args.start > 0:
        start_idx = args.start
    
    # Process the image list
    processed_count, total_tiles = process_image_list(imglist, start_idx, args.count)
    
    print(f"Done! Processed {processed_count} images ({total_tiles} tiles)")