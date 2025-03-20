#! /usr/bin/env/python3

"""
script to run a trained yolov8 segment model on unlabeled images, saving these results in cvat annotation form, used to bootstrap the cvat annotation process
NOTE: Basefile must have been downloaded from cvat, with the images already loaded into the job
"""

from ultralytics import YOLO
import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import zipfile
from Utils import classes, poly_2_rle
import sys

### File locations ###
base_file = "/home/java/Downloads/cgras_2024_empty/annotations.xml"
base_img_location = "/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/ultralytics_data"
output_filename = "/home/java/Downloads/cgras_2023_complete.xml"
# base_file = sys.args[1]
# base_img_location = sys.args[2]
# output_filename = sys.args[3]

### Parameters ###
weight_file = "/home/java/Java/ultralytics/runs/segment/train9/weights/cgras_yolov8n-seg_640p_20231209.pt"

class Detect2Cvat:
    BASE_FILE = "/home/java/Java/Cgras/cgras_settler_counter/annotations.xml"
    OUTPUT_FILE = "/home/java/Downloads/complete.xml"
    DEFAULT_WEIGHT_FILE = "/home/java/Java/ultralytics/runs/segment/train4/weights/best.pt"
    
    def __init__(self, 
                 img_location: str, 
                 output_file: str = OUTPUT_FILE, 
                 weights_file: str = DEFAULT_WEIGHT_FILE,
                 base_file: str = BASE_FILE, 
                 output_as_mask: str = False):
        self.img_location = img_location
        self.base_file = base_file
        self.output_file = output_file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = YOLO(weights_file).to(self.device)
        self.output_as_mask = output_as_mask


    def run(self):
        tree = ET.parse(self.base_file)
        root = tree.getroot() 
        new_tree = ElementTree(Element("annotations"))
        # add version element
        version_element = ET.Element('version')
        version_element.text = '1.1'
        new_tree.getroot().append(version_element)
        # add Meta elements, (copy over from source_file)
        meta_element = root.find('.//meta')
        if meta_element is not None:
            new_meta_elem = ET.SubElement(new_tree.getroot(), 'meta')
            # copy all subelements of meta
            for sub_element in meta_element:
                new_meta_elem.append(sub_element)
        

        for i, image_element in enumerate(root.findall('.//image')):
            print(i,'images being processed')
            image_id = image_element.get('id')
            image_name = image_element.get('name')
            image_width = int(image_element.get('width'))
            image_height = int(image_element.get('height'))

            # create new image element in new XML
            new_elem = SubElement(new_tree.getroot(), 'image')
            new_elem.set('id', image_id)
            new_elem.set('name', image_name)
            new_elem.set('width', str(image_width))
            new_elem.set('height', str(image_height))
            
            image_file = os.path.join(self.img_location, image_name)
            results = self.model.predict(source=image_file, iou=0.5, agnostic_nms=True)
            masks = results[0].masks
            class_list = [b.cls.item() for b in results[0].boxes]

            if masks==None:
                print('No masks found in image',image_name)
                continue

            for j, m in enumerate(masks):
                label = classes[int(class_list[j])]
                mxy = m.xy
                xy = np.squeeze(mxy)
                if self.output_as_mask:
                    try:
                        rle_string, left, top, width, height  = poly_2_rle(xy,", ",False)
                        mask_elem = SubElement(new_elem, 'mask')
                        mask_elem.set('label', label)
                        mask_elem.set('source', 'semi-auto')
                        mask_elem.set('occluded', '0')
                        mask_elem.set('rle', rle_string)
                        mask_elem.set('left', str(left))
                        mask_elem.set('top', str(top))
                        mask_elem.set('width', str(width))
                        mask_elem.set('height', str(height))
                        mask_elem.set('z_order', '0')
                    except:
                        print(f'mask {j} encountered problem xy = {xy}')
                else: 
                    if xy is None or len(xy)==0:
                        print(f'mask {j} encountered problem xy = {xy}')
                    else:
                        formatted_points = ';'.join([f"{x:.2f},{y:.2f}" for x, y in xy if x and y])
                        mask_elem = SubElement(new_elem, 'polygon')
                        mask_elem.set('label', label)
                        mask_elem.set('source', 'manual')
                        mask_elem.set('occluded', '0')
                        mask_elem.set('points', formatted_points)
                        mask_elem.set('z_order', '0')

            print(len(class_list),'masks converted in image',image_name)

        new_tree.write(self.output_file, encoding='utf-8', xml_declaration=True)

        zip_filename = self.output_file.split('.')[0] + '.zip'
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.output_file, arcname='output_xml_file.xml')
        print('XML file zipped')


print("Detecting corals and saving to annotation format.")
Det = Detect2Cvat(base_img_location, output_filename, weight_file, base_file)
Det.run()
print("Done detecting corals")
