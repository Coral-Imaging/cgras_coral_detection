import os
import json
from pathlib import Path
import cv2
from tqdm import tqdm

classes = ["recruit_live_white", "recruit_cluster_live_white", "recruit_symbiotic", "recruit_cluster_symbiotic", "recruit_partial",
           "recruit_cluster_partial", "recruit_dead", "recruit_cluster_dead", "grazer_snail", "pest_tubeworm", "unknown"] 


def process_labels_and_images(label_dir, image_dir, output_dir, output_json_path, ignore_classes=None):
    """
    Processes YOLO segmentation labels with polygon data to extract bounding boxes, crop images, 
    save them, and generate a JSON file for classification.

    Args:
        label_dir (str): Path to the folder containing segmentation label files.
        image_dir (str): Path to the folder containing corresponding images.
        output_dir (str): Directory to save cropped images for classification.
        output_json_path (str): Path to save the generated JSON file.
        ignore_classes (list): List of class indices to ignore.
    """
    if ignore_classes is None:
        ignore_classes = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store JSON data
    output_data = {}

    # Iterate through label files
    for label_file in tqdm(sorted(Path(label_dir).glob("*.txt")), desc="Processing labels"):
        # Parse corresponding image file
        image_name = label_file.stem + ".jpg"
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found. Skipping...")
            continue

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image {image_path}. Skipping...")
            continue

        img_height, img_width, _ = image.shape

        # Process label file
        with open(label_file, "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                # Parse label data
                parts = line.strip().split()
                class_id = int(parts[0])

                # Ignore specified classes
                if class_id in ignore_classes:
                    continue

                # Parse (x, y) coordinate pairs
                try:
                    coordinates = list(map(float, parts[1:]))
                    if len(coordinates) % 2 != 0:
                        print(f"Invalid number of coordinates in {label_file}: {line}")
                        continue

                    # Group coordinates into pairs
                    points = [(coordinates[j], coordinates[j + 1]) for j in range(0, len(coordinates), 2)]

                    # Calculate bounding box from polygon
                    x_coords = [p[0] * img_width for p in points]
                    y_coords = [p[1] * img_height for p in points]

                    x_min, x_max = max(0, int(min(x_coords))), min(img_width, int(max(x_coords)))
                    y_min, y_max = max(0, int(min(y_coords))), min(img_height, int(max(y_coords)))

                    # Skip if bounding box is invalid
                    if x_min >= x_max or y_min >= y_max:
                        print(f"Invalid bounding box in {label_file}: {line}")
                        continue

                    # Crop the image
                    cropped = image[y_min:y_max, x_min:x_max]

                    # Save cropped image
                    class_dir = os.path.join(output_dir, classes[class_id])
                    os.makedirs(class_dir, exist_ok=True)

                    crop_filename = f"{label_file.stem}_{i}_img.jpg"
                    crop_path = os.path.join(class_dir, crop_filename)
                    cv2.imwrite(crop_path, cropped)

                    # Add entry to JSON data
                    output_data[crop_filename] = {
                        "labels": [class_id],
                        "path": os.path.relpath(crop_path, output_dir)
                    }
                except ValueError as e:
                    print(f"Error parsing label file {label_file}: {e}")
                    continue

    # Save the JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    # Remove small images
    remove_small_images(output_dir)

    print(f"Processing complete. Cropped images saved in: {output_dir}")
    print(f"JSON file saved at: {output_json_path}")

def remove_small_images(output_dir, min_size_kb=2.5):
    """
    Removes images that are smaller than the specified size.

    Args:
        output_dir (str): Directory containing cropped images.
        min_size_kb (float): Minimum size in kilobytes. Images smaller than this will be removed.
    """
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) < min_size_kb * 1024:
                os.remove(file_path)
                print(f"Removed small image: {file_path}")

# Example usage
if __name__ == "__main__":
    label_dir = "/home/java/Java/hpc-home/Data/cgras/2240605_cgras/test/labels"  
    image_dir = "/home/java/Java/hpc-home/Data/cgras/2240605_cgras/test/images" 
    output_dir = "/home/java/Java/hpc-home/Data/cgras/classifier/classifer_split/test"  
    output_json_path = os.path.join(output_dir, "classifier_data.json")
    ignore_classes = [8, 9, 10]  # Example: ignore classes with indices 8 and 9
    process_labels_and_images(label_dir, image_dir, output_dir, output_json_path, ignore_classes)
