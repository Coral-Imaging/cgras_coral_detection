"""Convert CVAT1.1 to YOLO Segmentation (v8)
        - PDE export from CVAT was not working when exported as other fromats, therefore created code to change from one labeling format to another
        Works on masks only
"""

import xml.etree.ElementTree as ET
import numpy as np
from xml.etree.ElementTree import Element
from analysis.Utils import rle_to_binary_mask, classes, class_colours
import glob
import os
import cv2 as cv


classes = ["Platy", "Acro", "Recruit asymiobitc", "recruit asymiobitc cluster", "recruit symbiotic",
           "recruit symbiotic cluster", "recruit dead", "recruit dead cluster"]


def cvat_to_yolo(source_file: str, output_dir: str):
    """Converts a cvat1.1 source file annotation with rle to a folder of yolo .txt files
    NOTE: '.xml' must be included in the source_file and the output_dir should be a folder

    Args:
        source_file (str): name of source file with annontations
        output_dir (str): name of output folder to save to.
    '''"""
    os.makedirs(output_dir, exist_ok=True)
    tree = ET.parse(source_file)
    root = tree.getroot() 
    for i, image_element in enumerate(root.findall('.//image')):
        print(f"{i} of {len(root.findall('.//image'))} images being processed")
        image_id = image_element.get('id')
        image_name = image_element.get('name')

        image_width = int(image_element.get('width'))
        image_height = int(image_element.get('height'))

        txt_list_m = []
        for mask_ele in image_element.findall('mask'):
            txt_list = []
            class_name = mask_ele.get('label')
            if class_name in classes:
                class_idx = classes.index(class_name)
                txt_list.append(class_idx)
            mask_rle = list(map(int, mask_ele.get('rle').split(',')))
            mask_width, mask_height = int(mask_ele.get('width')), int(mask_ele.get('height'))
            mask_top, mask_left = int(mask_ele.get('top')), int(mask_ele.get('left'))
            try:
                mask = rle_to_binary_mask(mask_rle, mask_width, mask_height, SHOW_IMAGE=False)
            except:
                print("error in rle_to_binary_mask")
                import code
                code.interact(local=dict(globals(), **locals()))
            
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    print(f"Contour with insufficient points found for image {image_name}: {len(contour)} points")
                    continue
                points = np.squeeze(contour) + np.array([mask_left, mask_top])
                if len(points.shape) != 2 or points.shape[1] != 2:
                    print(f"Unexpected points shape for image {image_name}: {points.shape}")
                    continue
                normalized_points = np.column_stack((points[:, 0] / image_width, points[:, 1] / image_height))
                formatted_points_list = normalized_points.flatten().tolist()
                txt_list = txt_list + formatted_points_list
            txt_list_m.append(txt_list)
        txtsave_path = os.path.join(output_dir, image_name[:-4]  + '.txt')
        with open(txtsave_path, 'w') as file:
            for txt_result in txt_list_m:
                for item in txt_result:
                    file.write(str(item) + ' ')
                file.write('\n')

##CVAT mask (XML) to YOLO

if __name__ == "__main__":
    cvat_to_yolo(source_file="/home/java/Downloads/cvat_exportedascvat.xml", output_dir='/home/java/Downloads/cvat_yolov8seg_pde/label')