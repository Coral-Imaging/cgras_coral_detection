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

    def __init__(self, full_image_dir):
        """
        :param full_image_dir: Path to the directory containing the full-size images.
        """

        self.full_image_dir = full_image_dir

    def extract_tile_info(self, tile_filename):
        """
        Extracts the original image name and tile coordinates from the tile filename.
        :param tile_filename: Name of the tile image file (e.g., "CGRAS_Amag_MIS5a_2024112_w2_T01_06_6720_2880.jpg").
        :return: Tuple (original_image_name, x_start, y_start) or None if parsing fails.
        """
        match = re.match(r"(.+)_([0-9]+)_([0-9]+)\.jpg", tile_filename)
        if match:
            base_name, x_start, y_start = match.groups()
            return base_name, int(x_start), int(y_start)
        return None

    def find_and_show_tile(self, tile_path, full_image_dir=None):
        """
        Finds the corresponding full image and highlights the tile region.
        :param tile_path: Path to the tiled image file.
        :param full_image_dir: Optional override of full image directory.
        """
        if full_image_dir is not None:
            self.full_image_dir = full_image_dir

        tile_filename = os.path.basename(tile_path)
        tile_info = self.extract_tile_info(tile_filename)
        
        if not tile_info:
            print(f"Invalid tile filename format: {tile_filename}")
            return

        base_name, x_start, y_start = tile_info
        full_image_path = os.path.join(self.full_image_dir, f"{base_name}.jpg")

        if not os.path.exists(full_image_path):
            print(f"Full image not found: {full_image_path}")
            return

        # Load full image
        full_img = cv.imread(full_image_path)
        full_img = cv.cvtColor(full_img, cv.COLOR_BGR2RGB)

        # Determine tile size (assuming all tiles are the same size)
        tile_img = cv.imread(tile_path)
        tile_height, tile_width = tile_img.shape[:2]

        # Draw the rectangle on the original image
        cv.rectangle(full_img, (x_start, y_start), 
                     (x_start + tile_width, y_start + tile_height), 
                     (255, 0, 0), 3)

        # Display the result
        plt.figure(figsize=(10, 10))
        plt.imshow(full_img)
        plt.title(f"Tile {tile_filename} in Original Image")
        plt.axis("off")
        plt.show()

    def show_tile_with_labels(self, tile_path):
        """
        Loads the tile image and overlays ground truth annotations from the corresponding label file.
        :param tile_path: Path to the tiled image file.
        """
        tile_filename = os.path.basename(tile_path)
        tile_info = self.extract_tile_info(tile_filename)
        
        if not tile_info:
            print(f"Invalid tile filename format: {tile_filename}")
            return

        base_name, x_start, y_start = tile_info

        # Determine label file path
        label_path = tile_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            return

        # Load tile image
        tile_img = cv.imread(tile_path)
        tile_img = cv.cvtColor(tile_img, cv.COLOR_BGR2RGB)
        tile_height, tile_width = tile_img.shape[:2]

        # Read label file and draw polygons
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

        # Display the result with a legend
        plt.figure(figsize=(6, 6))
        plt.imshow(tile_img)
        plt.title(f"Tile with Ground Truth: {tile_filename}")
        plt.axis("off")

        # Add legend
        legend_patches = [plt.Line2D([0], [0], color=np.array(c)/255, lw=4, label=name) 
                          for name, c in self.CLASS_COLORS.items()]
        plt.legend(handles=legend_patches, loc="upper right", fontsize=10)

        plt.show()

    def show_full_image_with_labels(self, full_image_path):
        """
        Displays the full image with its corresponding ground truth labels.
        :param full_image_path: Path to the full-size image file.
        """
        base_name = os.path.basename(full_image_path).replace(".jpg", "")
        label_path = full_image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            return

        # Load full image
        full_img = cv.imread(full_image_path)
        full_img = cv.cvtColor(full_img, cv.COLOR_BGR2RGB)
        img_height, img_width = full_img.shape[:2]

        # Read label file and draw polygons
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

        # Display the result with a legend
        plt.figure(figsize=(12, 12))
        plt.imshow(full_img)
        plt.title(f"Full Image with Ground Truth: {base_name}")
        plt.axis("off")

        # Add legend
        legend_patches = [plt.Line2D([0], [0], color=np.array(c)/255, lw=4, label=name) 
                        for name, c in self.CLASS_COLORS.items()]
        plt.legend(handles=legend_patches, loc="upper right", fontsize=12)

        plt.show()


# Example usage
if __name__ == '__main__':
    full_image_dir = "/media/agoni/RRAP03/exported_labelled_from_cvat/exported_2024_cgras_amag_T01_first10_100quality/data/images/Train"
    full_image_path = "/media/agoni/RRAP03/exported_labelled_from_cvat/exported_2024_cgras_amag_T01_first10_100quality/data/images/Train/CGRAS_Amag_MIS5a_20241112_w2_T01_06.jpg"
    tile_image_path = "/media/agoni/RRAP03/outputs/image_tiler/images/CGRAS_Amag_MIS5a_20241112_w2_T01_06_6720_2880.jpg"

    finder = FindTileInImage(full_image_dir)
    finder.find_and_show_tile(tile_image_path)
    finder.show_full_image_with_labels(full_image_path)
    finder.show_tile_with_labels(tile_image_path)
