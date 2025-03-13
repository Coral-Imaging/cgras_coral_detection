import os
import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class FindTileInImage:
    CLASS_LABELS = ["alive", "dead", "mask_alive", "mask_dead"]
    CLASS_COLORS = {
        "alive": (0, 255, 0),       # Green
        "dead": (255, 0, 0),        # Red
        "mask_alive": (0, 255, 255), # Yellow
        "mask_dead": (255, 165, 0)   # Orange
    }

    def __init__(self, ssd_path):
        """
        Initialize with the SSD base path.
        """
        self.ssd_path = ssd_path
        self.base_dir = os.path.join(ssd_path, "cgras_2024_aims_camera_trolley")
        self.tiled_image_dir = os.path.join(ssd_path, "outputs/tiled_images")

    def find_file(self, filename):
        """
        Recursively search for a file within the base directory.
        """
        for root, _, files in os.walk(self.base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def extract_tile_info(self, tile_filename):
        """
        Extracts the original image name and tile coordinates from the tile filename.
        """
        match = re.match(r"(.+)_([0-9]+)_([0-9]+)\\.jpg", tile_filename)
        if match:
            base_name, x_start, y_start = match.groups()
            return base_name, int(x_start), int(y_start)
        return None

    def find_and_show_tile(self, full_image_name, tile_image_name):
        """
        Locate and display the tile within the corresponding full image.
        """
        full_image_path = self.find_file(full_image_name)
        tile_image_path = self.find_file(tile_image_name)

        if not full_image_path or not tile_image_path:
            print("Error: File(s) not found.")
            return

        tile_filename = os.path.basename(tile_image_path)
        tile_info = self.extract_tile_info(tile_filename)
        
        if not tile_info:
            print(f"Invalid tile filename format: {tile_filename}")
            return

        base_name, x_start, y_start = tile_info

        # Load images
        full_img = cv.imread(full_image_path)
        full_img = cv.cvtColor(full_img, cv.COLOR_BGR2RGB)
        tile_img = cv.imread(tile_image_path)
        tile_height, tile_width = tile_img.shape[:2]

        # Draw rectangle
        cv.rectangle(full_img, (x_start, y_start), 
                     (x_start + tile_width, y_start + tile_height), 
                     (255, 0, 0), 3)

        # Display image
        plt.figure(figsize=(10, 10))
        plt.imshow(full_img)
        plt.title(f"Tile {tile_filename} in Original Image")
        plt.axis("off")
        plt.show()

    def show_tile_with_labels(self, tile_image_name):
        """
        Loads the tile image and overlays ground truth annotations from the corresponding label file.
        """
        tile_image_path = self.find_file(tile_image_name)
        if not tile_image_path:
            print("Tile image not found.")
            return

        tile_filename = os.path.basename(tile_image_path)
        tile_info = self.extract_tile_info(tile_filename)
        if not tile_info:
            print(f"Invalid tile filename format: {tile_filename}")
            return

        label_path = tile_image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            return

        tile_img = cv.imread(tile_image_path)
        tile_img = cv.cvtColor(tile_img, cv.COLOR_BGR2RGB)
        tile_height, tile_width = tile_img.shape[:2]

        with open(label_path, "r") as label_file:
            lines = label_file.readlines()

        for line in lines:
            parts = line.strip().split()
            class_idx = int(parts[0])
            points = [float(p) for p in parts[1:]]
            abs_points = np.array([
                (int(points[i] * tile_width), int(points[i + 1] * tile_height))
                for i in range(0, len(points), 2)
            ], np.int32)

            if len(abs_points) > 0:
                class_name = self.CLASS_LABELS[class_idx]
                color = self.CLASS_COLORS[class_name]
                cv.polylines(tile_img, [abs_points], isClosed=True, color=color, thickness=2)
                cv.putText(tile_img, class_name, (abs_points[0][0], abs_points[0][1]), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        plt.figure(figsize=(6, 6))
        plt.imshow(tile_img)
        plt.title(f"Tile with Ground Truth: {tile_filename}")
        plt.axis("off")
        plt.show()

    def show_full_image_with_labels(self, full_image_name):
        """
        Displays the full image with its corresponding ground truth labels.
        """
        full_image_path = self.find_file(full_image_name)
        if not full_image_path:
            print("Full image not found.")
            return

        label_path = full_image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            return

        full_img = cv.imread(full_image_path)
        full_img = cv.cvtColor(full_img, cv.COLOR_BGR2RGB)
        img_height, img_width = full_img.shape[:2]

        with open(label_path, "r") as label_file:
            lines = label_file.readlines()

        for line in lines:
            parts = line.strip().split()
            class_idx = int(parts[0])
            points = [float(p) for p in parts[1:]]
            abs_points = np.array([
                (int(points[i] * img_width), int(points[i + 1] * img_height))
                for i in range(0, len(points), 2)
            ], np.int32)

            if len(abs_points) > 0:
                class_name = self.CLASS_LABELS[class_idx]
                color = self.CLASS_COLORS[class_name]
                cv.polylines(full_img, [abs_points], isClosed=True, color=color, thickness=2)
                cv.putText(full_img, class_name, (abs_points[0][0], abs_points[0][1]), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        plt.figure(figsize=(12, 12))
        plt.imshow(full_img)
        plt.title(f"Full Image with Ground Truth: {full_image_name}")
        plt.axis("off")
        plt.show()


# Example usage
if __name__ == '__main__':

    path_to_folder = "/media/java/RRAP03/"

    full_image_name = "CGRAS_Amag_MIS5a_20241112_w2_T01_06.jpg"
    tile_image_name = "CGRAS_Amag_MIS5a_20241112_w2_T01_06_6720_2880.jpg"

    finder = FindTileInImage(path_to_folder)

    finder.find_and_show_tile(full_image_name, tile_image_name)
    finder.show_full_image_with_labels(full_image_name)
    finder.show_tile_with_labels(tile_image_name)
