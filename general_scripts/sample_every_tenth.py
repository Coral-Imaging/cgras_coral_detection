#!/usr/bin/env python3
"""
Script to sample every tenth image/label pair from a dataset folder.
Maintains correspondence between images and labels.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import glob

def get_corresponding_files(images_dir, labels_dir, image_extensions=None, label_extension='.txt'):
    """
    Get corresponding image and label files.
    
    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory
        image_extensions: List of image extensions to look for
        label_extension: Extension for label files
    
    Returns:
        List of tuples (image_path, label_path) for corresponding files
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(images_dir, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern = os.path.join(images_dir, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # Sort image files
    image_files = sorted(image_files)
    
    # Find corresponding label files
    corresponding_pairs = []
    for image_path in image_files:
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}{label_extension}")
        
        if os.path.exists(label_path):
            corresponding_pairs.append((image_path, label_path))
        else:
            print(f"Warning: No corresponding label found for {image_path}")
    
    return corresponding_pairs

def sample_every_nth(file_pairs, n=10, start_index=0):
    """
    Sample every nth file pair starting from start_index.
    
    Args:
        file_pairs: List of (image_path, label_path) tuples
        n: Sample every nth file (default: 10)
        start_index: Starting index (default: 0)
    
    Returns:
        List of sampled file pairs
    """
    return file_pairs[start_index::n]

def copy_files_to_new_folder(file_pairs, output_dir, preserve_structure=True):
    """
    Copy image and label pairs to a new folder structure.
    
    Args:
        file_pairs: List of (image_path, label_path) tuples
        output_dir: Output directory path
        preserve_structure: Whether to create images/labels subdirectories
    """
    output_path = Path(output_dir)
    
    if preserve_structure:
        images_output = output_path / 'images'
        labels_output = output_path / 'labels'
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    failed_pairs = []
    successful_images = []
    successful_labels = []
    
    for image_path, label_path in file_pairs:
        image_copied = False
        label_copied = False
        
        try:
            if preserve_structure:
                # Copy to images and labels subdirectories
                shutil.copy2(image_path, images_output)
                image_copied = True
                successful_images.append(os.path.basename(image_path))
                
                shutil.copy2(label_path, labels_output)
                label_copied = True
                successful_labels.append(os.path.basename(label_path))
            else:
                # Copy directly to output directory
                shutil.copy2(image_path, output_path)
                image_copied = True
                successful_images.append(os.path.basename(image_path))
                
                shutil.copy2(label_path, output_path)
                label_copied = True
                successful_labels.append(os.path.basename(label_path))
            
            copied_count += 1
            print(f"Copied: {os.path.basename(image_path)} and {os.path.basename(label_path)}")
            
        except Exception as e:
            print(f"Error copying {image_path} or {label_path}: {e}")
            failed_pairs.append((image_path, label_path, str(e)))
            
            # If one file copied but the other failed, we need to clean up
            if image_copied and not label_copied:
                try:
                    if preserve_structure:
                        os.remove(images_output / os.path.basename(image_path))
                    else:
                        os.remove(output_path / os.path.basename(image_path))
                    successful_images.pop()  # Remove from success list
                    print(f"Cleaned up orphaned image: {os.path.basename(image_path)}")
                except:
                    pass
    
    # Validation check
    if len(successful_images) != len(successful_labels):
        print(f"WARNING: Mismatch in copied files!")
        print(f"  Images copied: {len(successful_images)}")
        print(f"  Labels copied: {len(successful_labels)}")
        
        # Try to identify the mismatch
        img_basenames = set(os.path.splitext(name)[0] for name in successful_images)
        lbl_basenames = set(os.path.splitext(name)[0] for name in successful_labels)
        
        missing_labels = img_basenames - lbl_basenames
        missing_images = lbl_basenames - img_basenames
        
        if missing_labels:
            print(f"  Images without labels: {missing_labels}")
        if missing_images:
            print(f"  Labels without images: {missing_images}")
    else:
        print(f"✓ Validation passed: {len(successful_images)} image/label pairs copied successfully")
    
    if failed_pairs:
        print(f"\nFailed to copy {len(failed_pairs)} pairs:")
        for img, lbl, error in failed_pairs:
            print(f"  {os.path.basename(img)} + {os.path.basename(lbl)}: {error}")
    
    print(f"\nTotal files copied: {copied_count} pairs ({copied_count * 2} files)")
    
    return successful_images, successful_labels, failed_pairs

def validate_output_folder(output_dir, preserve_structure=True):
    """
    Validate that the output folder has matching numbers of images and labels.
    
    Args:
        output_dir: Output directory to validate
        preserve_structure: Whether images/labels are in subdirectories
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    output_path = Path(output_dir)
    
    if preserve_structure:
        images_dir = output_path / 'images'
        labels_dir = output_path / 'labels'
    else:
        images_dir = output_path
        labels_dir = output_path
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: Required directories don't exist")
        return False
    
    # Get image files
    image_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
        image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
    
    # Get label files
    label_files = list(labels_dir.glob("*.txt"))
    
    # Get base names (without extensions)
    image_basenames = set(f.stem for f in image_files)
    label_basenames = set(f.stem for f in label_files)
    
    print(f"Validation results:")
    print(f"  Images found: {len(image_files)}")
    print(f"  Labels found: {len(label_files)}")
    print(f"  Matching pairs: {len(image_basenames & label_basenames)}")
    
    # Check for mismatches
    missing_labels = image_basenames - label_basenames
    missing_images = label_basenames - image_basenames
    
    if missing_labels:
        print(f"  Images without labels: {missing_labels}")
    if missing_images:
        print(f"  Labels without images: {missing_images}")
    
    is_valid = len(missing_labels) == 0 and len(missing_images) == 0
    
    if is_valid:
        print(f"✓ Validation PASSED: All files have matching pairs")
    else:
        print(f"✗ Validation FAILED: Mismatched files found")
    
    return is_valid

def main():
    parser = argparse.ArgumentParser(description='Sample every nth image/label pair from a dataset')
    parser.add_argument('input_dir', nargs='?', help='Input directory containing images and labels subdirectories')
    parser.add_argument('output_dir', nargs='?', help='Output directory for sampled files')
    parser.add_argument('--nth', type=int, default=10, help='Sample every nth file (default: 10)')
    parser.add_argument('--start', type=int, default=0, help='Starting index (default: 0)')
    parser.add_argument('--images-subdir', default='images', help='Name of images subdirectory (default: images)')
    parser.add_argument('--labels-subdir', default='labels', help='Name of labels subdirectory (default: labels)')
    parser.add_argument('--label-ext', default='.txt', help='Label file extension (default: .txt)')
    parser.add_argument('--no-structure', action='store_true', help='Don\'t preserve images/labels folder structure')
    parser.add_argument('--image-exts', nargs='+', default=['.jpg', '.jpeg', '.png', '.tif', '.tiff'], 
                        help='Image file extensions to look for')
    parser.add_argument('--validate-only', help='Only validate an existing output directory (provide path)')
    
    args = parser.parse_args()
    
    # Handle validation-only mode
    if args.validate_only:
        preserve_structure = not args.no_structure
        is_valid = validate_output_folder(args.validate_only, preserve_structure)
        return 0 if is_valid else 1
    
    # Validate required arguments for sampling mode
    if not args.input_dir or not args.output_dir:
        parser.error("input_dir and output_dir are required for sampling mode")
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return
    
    images_dir = input_path / args.images_subdir
    labels_dir = input_path / args.labels_subdir
    
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        return
    
    if not labels_dir.exists():
        print(f"Error: Labels directory {labels_dir} does not exist")
        return
    
    print(f"Looking for corresponding files in:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    
    # Get corresponding file pairs
    file_pairs = get_corresponding_files(images_dir, labels_dir, args.image_exts, args.label_ext)
    
    if not file_pairs:
        print("No corresponding image/label pairs found!")
        return
    
    print(f"Found {len(file_pairs)} corresponding image/label pairs")
    
    # Sample every nth pair
    sampled_pairs = sample_every_nth(file_pairs, args.nth, args.start)
    
    print(f"Sampling every {args.nth} files starting from index {args.start}")
    print(f"Selected {len(sampled_pairs)} pairs for copying")
    
    if len(sampled_pairs) == 0:
        print("No files to copy!")
        return
    
    # Copy files
    preserve_structure = not args.no_structure
    successful_images, successful_labels, failed_pairs = copy_files_to_new_folder(
        sampled_pairs, args.output_dir, preserve_structure
    )
    
    print(f"\nSampling complete! Output saved to: {args.output_dir}")
    
    # Final validation report
    if len(successful_images) == len(successful_labels):
        print(f"✓ Success: {len(successful_images)} complete image/label pairs copied")
    else:
        print(f"⚠ Warning: Mismatch in final counts!")
        print(f"  Images: {len(successful_images)}")
        print(f"  Labels: {len(successful_labels)}")
        
    if failed_pairs:
        print(f"⚠ {len(failed_pairs)} pairs failed to copy - see details above")
    
    # Run final validation
    print(f"\nRunning final validation...")
    final_validation = validate_output_folder(args.output_dir, preserve_structure)
    
    if final_validation:
        print(f"🎉 Sampling completed successfully with validation!")
    else:
        print(f"⚠ Sampling completed but validation found issues!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
