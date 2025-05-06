#! /usr/bin/env python3

""" pde_to_yolo.py
    Convert PDE annotations from CSV files to YOLO format
    The class information was stored in the CSV file an the mask data were stored in ROIs, and was extract and converted to CVAT format for further labeling
    Used in combination with ROI_2
"""
import os
import pandas as pd
import numpy as np
import cv2 as cv
import glob
import re
import math
from analysis.utils.Utils import poly_2_rle
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
import zipfile
import sys


## File locations ###
base_file = "/home/java/Downloads/pde2/annotations.xml"
output_filename = "/home/java/Downloads/pde2.xml"
#segment_filename = f"{os.path.splitext(filename)[0]}_segment_{left}_{upper}_{right}_{lower}.jpg"
img_path= '/media/java/cslics_ssd1/SCU_Pdae_Data/all_jpg'
label_path = '/media/java/cslics_ssd1/SCU_Pdae_Data/ROIs Extracted'
imgsave_path = '/media/java/cslics_ssd1/SCU_Pdae_Data/annotated'
img_list = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
want_box = False
SAVE = True #save image cut up with annotations label

def contained_within(box1, box2):
    """Check if box1 is contained within box2"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return x1 >= x3 and y1 >= y3 and x2 <= x4 and y2 <= y4

tree = ET.parse(base_file)
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
    
    image_file = os.path.join(img_path, image_name)
    image_left = int(image_name.split('_')[2])
    image_upper = int(image_name.split('_')[3])
    image_right = int(image_name.split('_')[4])
    image_lower = int(image_name.split('_')[5][:-4])

    if image_name.split('_')[0].split(' ')[0] == 'T23':
        im_folder = image_name.split('_')[0].split(' ')[0]+' (Done)'
    else:
        im_folder = image_name.split('_')[0].split(' ')[0][:-1]+'0'+image_name.split('_')[0].split(' ')[0][-1]+' (Done)'
    true_label_name = image_name.split('_')[0]+'_ROI_Extract.csv'
    if image_name.split('_')[0][-1] == ' ' and image_name.split('_')[0].split(' ')[0] == 'T6':
        true_label_name = image_name.split('_')[0][:-1]+'_ROI_Extract.csv'
    if image_name.split('_')[0].split(' ')[0] != 'T3':
        print(f"Skipping {image_name}")
        continue
    cvimg = cv.imread(image_file)
    masked = cvimg.copy()

    labels = os.path.join(label_path, im_folder, true_label_name)
    try:
        df = pd.read_csv(labels, delimiter='\t')
    except:
        print(f"Error reading {labels}")
        import code
        code.interact(local={**locals(), **globals()})

    masks_in = 0
    for index, row in df.iterrows():
        id = row['name']
        type = row['type']
        if type=='Platy' or type=='Platy ':
            class_name = "Platy"
        else:
            class_name = "Acro"
        top = row['top']
        left = row['left']
        bottom = row['bottom']
        right = row['right']
        n_coordinates = row['n_coordinates']
        integer_coordinates = row['integer_coordinates']
        subpixel_coordinates = row['subpixel_coordinates']

        if not contained_within((left, top, right, bottom), (image_left, image_upper, image_right, image_lower)):
            continue
        masks_in += 1
        cords = []
        try:
            for i in subpixel_coordinates.split('\n'):
                matches = re.findall(r'[-+]?\d*\.\d+|\d+', i)
                if len(matches) == 2:
                    y = float(matches[1])
                    x = float(matches[0])
                    cords.append((x, y))
        except Exception as e:
            print(f"Error drawing polygons for {image_name}: {e}")
            continue
        if want_box:
            # create new box element in new XML
            box_elem = SubElement(new_elem, 'box')
            box_elem.set('label', class_name)
            box_elem.set('occluded', '0')
            box_elem.set('xtl', str(left - image_left))
            box_elem.set('ytl', str(top - image_upper))
            box_elem.set('xbr', str(right - image_left))
            box_elem.set('ybr', str(bottom - image_upper))
            box_elem.set('z_order', '0')
        # create new polygon element in new XML
        points = np.array(cords, np.int32).reshape(-1,2)
        shifted_points = points - [image_left, image_upper]
        
        if SAVE:
            desired_color = (0, 255, 0)
            cv.fillPoly(masked, [shifted_points], desired_color)
        rle, rleft, rtop, rwidth, rheight = poly_2_rle(shifted_points, ", ", False)
        mask_elem = SubElement(new_elem, 'mask')
        mask_elem.set('label', class_name)
        mask_elem.set('source', 'semi-auto')
        mask_elem.set('occluded', '0')
        mask_elem.set('rle', rle)
        mask_elem.set('left', str(rleft))
        mask_elem.set('top', str(rtop))
        mask_elem.set('width', str(rwidth))
        mask_elem.set('height', str(rheight))
        mask_elem.set('z_order', '0')  
    if SAVE:
        alpha = 0.5
        semi_transparent_mask = cv.addWeighted(cvimg, 1-alpha, masked, alpha, 0)
        imgsavename = os.path.join(imgsave_path, image_name)    
        cv.imwrite(imgsavename, semi_transparent_mask)
        

new_tree.write(output_filename, encoding='utf-8', xml_declaration=True)


import code
code.interact(local={**locals(), **globals()})

## test without CVAT (save and visualise using CV)
for i, img in enumerate(img_list):
    print(f'Processing {i} of {len(img_list)}: {img}')

    image = cv.imread(img)
    masked = image.copy()
    img_name = os.path.basename(img)
    image_left = int(img_name.split('_')[2])
    image_upper = int(img_name.split('_')[3])
    image_right = int(img_name.split('_')[4])
    image_lower = int(img_name.split('_')[5][:-4])

    labels = os.path.join(label_path, img_name.split('_')[0]+'_ROI_Extract.csv')
    df = pd.read_csv(labels, delimiter='\t')

    counts = [0, 0]
    for index, row in df.iterrows():
        id = row['name']
        type = row['type']
        top = row['top']
        left = row['left']
        bottom = row['bottom']
        right = row['right']
        n_coordinates = row['n_coordinates']
        integer_coordinates = row['integer_coordinates']
        subpixel_coordinates = row['subpixel_coordinates']

        if not contained_within((left, top, right, bottom), (image_left, image_upper, image_right, image_lower)):
            counts[0] += 1
            continue
        counts[1] += 1
        cords = []
        try:
            for i in subpixel_coordinates.split('\n'):
                matches = re.findall(r'[-+]?\d*\.\d+|\d+', i)
                if len(matches) == 2:
                    y = float(matches[1])
                    x = float(matches[0])
                    cords.append((x, y))
        except Exception as e:
            print(f"Error drawing polygons for {img_name}: {e}")
            continue
        
        shifted_box = [left - image_left, top - image_upper, right-image_left, bottom-image_upper]
        cv.rectangle(image, (shifted_box[0], shifted_box[1]), (shifted_box[2], shifted_box[3]), (255, 0, 0), 2)
        cv.putText(image, type, (shifted_box[0], shifted_box[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        #polygons
        desired_color = (0, 255, 0)
        points = np.array(cords, np.int32).reshape(-1,2)
        shifted_points = points - [image_left, image_upper]
        cv.fillPoly(masked, [points], desired_color)
        
    print(f"Total annotations: {counts[0] + counts[1]}, Annotations within image: {counts[1]}")
    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.join(imgsave_path, img_name[:-4] + '.jpg')    
    cv.imwrite(imgsavename, semi_transparent_mask)


    import code
    code.interact(local={**locals(), **globals()})

