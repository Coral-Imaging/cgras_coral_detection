#! /usr/bin/env/python3
#TODO convert this to classes and config files
#
"""
Functions and varibles used for annotation and convertion to and from CVAT annotation style 
As well as helper functions for viewing predictions and robowflow sahi.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import supervision as sv

classes = ["alive_coral", "dead_coral", "recruit_symbiotic", "recruit_cluster_symbiotic", "recruit_partial",
           "recruit_cluster_partial", "recruit_dead", "recruit_cluster_dead", "grazer_snail", "pest_tubeworm", "unknown"] #how its in cvat

# Colours for each class
orange = [255, 128, 0] 
blue = [0, 212, 255] 
purple = [170, 0, 255] 
yellow = [255, 255, 0] 
brown = [144, 65, 2] 
green = [0, 255, 00] 
red = [255, 0, 0]
cyan = [0, 255, 255]
dark_purple =  [128, 0, 128]
light_grey =  [192, 192, 192] 
dark_green = [0, 100, 0] 
class_colours = {classes[0]: blue,
                classes[1]: green,
                classes[2]: purple,
                classes[3]: yellow,
                classes[4]: brown,
                classes[5]: cyan,
                classes[6]: orange,
                classes[7]: red,
                classes[8]: dark_purple,
                classes[9]: light_grey,
                classes[10]: dark_green}

def binary_mask_to_rle(binary_mask):
    """
    Convert a binary np array into a RLE format.
    Args:
        binary_mask (uint8 2D numpy array): binary mask
    Returns:
        rle: list of rle numbers
    """
    # Flatten the binary mask and append a zero at the end to handle edge case
    flat_mask = binary_mask.flatten()
    # Find the positions where the values change
    changes = np.diff(flat_mask)
    # Get the run lengths
    runs = np.where(changes != 0)[0] + 1
    # Get the lengths of each run
    run_lengths = np.diff(np.concatenate([[0], runs]))
    return run_lengths.tolist()

def rle_to_binary_mask(rle_list, 
                       width: int, 
                       height: int, 
                       SHOW_IMAGE: bool):
    """rle_to_binary_mask
    Converts a rle_list into a binary np array. Used to check the binary_mask_to_rle function

    Args:
        rle_list (list of strings): containing the rle information
        width (int): width of shape
        height (int): height of shape
        SHOW_IMAGE (bool): True if binary mask wants to be viewed

    Returns:
        mask: uint8 2D np array
    """
    mask = np.zeros((height, width), dtype=np.uint8) 
    current_pixel = 0
    
    for i in range(0, len(rle_list)):
        run_length = int(rle_list[i]) #find the length of current 0 or 1 run
        if (i % 2 == 0): #if an even number the pixel value will be 0
            run_value = 0
        else:
            run_value = 1

        for j in range(run_length): #fill the pixel with the correct value
            mask.flat[current_pixel] = run_value 
            current_pixel += 1

    if (SHOW_IMAGE):
        print("rle_list to binary mask")
        plt.imshow(mask, cmap='binary')
        plt.show()

    return mask


def poly_2_rle(points, 
               str_join: str,
               SHOW_IMAGE: bool):
    """poly_2_rle
    Converts a set of points for a polygon into an rle string and saves the data

    Args:
        points (2D numpy array): points of polygon
        str_join: how the points should be joined together
        SHOW_IMAGE (bool): True if binary mask wants to be viewed
    
    Returns:
        rle_string: string of the rle numbers,
        left: (int) left positioning of the rle numbers in pixles,
        top: (int) top positioning of the rle numbers in pixles,
        width: (int) width of the bounding box of the rle numbers in pixels,
        height: (int) height of the bounding box of the rle numbers in pixles
    """
    # create bounding box
    left = int(np.min(points[:, 0]))
    top = int(np.min(points[:, 1]))
    right = int(np.max(points[:, 0]))
    bottom = int(np.max(points[:, 1]))
    width = right - left + 1
    height = bottom - top + 1

    # create mask size of bounding box
    mask = np.zeros((height, width), dtype=np.uint8)
    # points relative to bounding box
    points[:, 0] -= left
    points[:, 1] -= top
    # fill mask where points are
    cv.fillPoly(mask, [points.astype(int)], color=1)

    # visual check of mask - looks good
    #SHOW_IMAGE = False
    if (SHOW_IMAGE):
        plt.imshow(mask, cmap='binary')
        plt.show()

    # convert the mask into a rle
    mask_rle = binary_mask_to_rle(mask)

    # rle string
    rle_string = str_join.join(map(str, mask_rle))
    
    return rle_string, left, top, width, height

def overlap_boxes(box1, box2):
    """
    Check if two axis-aligned bounding boxes overlap.

    Args:
        box1 (tuple or list): Coordinates of the first box in the form (x1, y1, x2, y2), 
                              where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
        box2 (tuple or list): Coordinates of the second box in the same format as box1.

    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    if x1 > x4 or x3 > x2:
        return False
    if y1 > y4 or y3 > y2:
        return False
    return True

def callback(image_slice: np.ndarray) -> sv.Detections:
    """
    Callback function for supervision slicer
    Args:
        image_slice: np.ndarray: a sliced image
    Returns:
        detections: supervisoon generated detections
    """
    results = model(image_slice)
    try:
        detections = sv.Detections.from_ultralytics(results[0])
    except:
        print("Error in callback")
        import code
        code.interact(local=dict(globals(), **locals()))
    return detections

def combine_2_annotations(box1, box2, cls_id_list, i, j, mask1, mask2):
    """Combines 2 annotations that are overlaping
    Args:
        box1 (list): Bounding box coordinates of the first annotation [x1, y1, x2, y2].
        box2 (list): Bounding box coordinates of the second annotation [x1, y1, x2, y2].
        cls_id_list (list): List of class IDs for all annotations.
        i (int): Index of the first annotation in the list.
        j (int): Index of the second annotation in the list.
        mask1 (tuple): Mask data for the first annotation, represented as 
                       (mask_array, top_left_x, top_left_y, width, height).
        mask2 (tuple): Mask data for the second annotation, represented as 
                       (mask_array, top_left_x, top_left_y, width, height).

    Returns:
        tuple: Combined annotation containing:
            - new_box (list): Combined bounding box [x1, y1, x2, y2].
            - new_class (int): Class ID of the combined annotation.
            - new_conf (float): Confidence score of the combined annotation.
            - new_mask (tuple): Combined mask data as (mask_array, top_left_x, top_left_y, width, height).
    """
    new_box = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]
    new_class = cls_id_list[i] if conf_list[i] > conf_list[j] else cls_id_list[j]
    new_conf = (conf_list[i] + conf_list[j]) / 2
    mask1_tl_x, mask1_tl_y, mask1_w, mask1_h = mask1[1], mask1[2], mask1[3], mask1[4]
    mask2_tl_x, mask2_tl_y, mask2_w, mask2_h = mask2[1], mask2[2], mask2[3], mask2[4]
    # New mask
    new_tl_x, new_tl_y = min(mask1_tl_x, mask2_tl_x), min(mask1_tl_y, mask2_tl_y)
    new_w = max(mask1_tl_x + mask1_w, mask2_tl_x + mask2_w) - new_tl_x
    new_h = max(mask1_tl_y + mask1_h, mask2_tl_y + mask2_h) - new_tl_y
    new_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    mask1_x_offset = mask1_tl_x - new_tl_x
    mask1_y_offset = mask1_tl_y - new_tl_y
    new_mask[mask1_y_offset:mask1_y_offset + mask1_h, mask1_x_offset:mask1_x_offset + mask1_w] = mask1[0]
    mask2_x_offset = mask2_tl_x - new_tl_x
    mask2_y_offset = mask2_tl_y - new_tl_y
    new_mask[mask2_y_offset:mask2_y_offset + mask2_h, mask2_x_offset:mask2_x_offset + mask2_w] = np.logical_or(
        new_mask[mask2_y_offset:mask2_y_offset + mask2_h, mask2_x_offset:mask2_x_offset + mask2_w], mask2[0]
    ).astype(np.uint8)
    return new_box, new_class, new_cof, new_mask

def combine_detections(box_array, conf_list, cls_id_list, mask_list):
    """
    Combine overlapping detections into a singular mask.

    Args:
        box_array (list or np.ndarray): Array of bounding box coordinates in the form [[x1, y1, x2, y2], ...],
                                        where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
        conf_list (list): List of confidence scores corresponding to each detection.
        cls_id_list (list): List of class IDs corresponding to each detection.
        mask_list (list): List of masks corresponding to each detection. Each mask is represented as a tuple:
                          (mask_array, top_left_x, top_left_y, width, height).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Updated array of combined bounding box coordinates.
            - list: Updated list of confidence scores.
            - list: Updated list of class IDs.
            - list: Updated list of combined masks.
    """
    updated_box_array, updated_conf_list, updated_class_id, updated_mask_list = [], [], [], []
    combined_indices = set()
    for i, mask1 in enumerate(mask_list):
        if i in combined_indices:
            continue  # Skip already combined detections
        box1 = box_array[i]
        overlap = False
        for j in range(i + 1, len(mask_list)):
            if j in combined_indices or j<i:
                continue
            mask2 = mask_list[j]
            box2 = box_array[j]
            if overlap_boxes(box1, box2): #assume only pairs overlap
                overlap = True
                new_box, new_class, new_cof, new_mask = combine_2_annotations(box1, box2, cls_id_list, i, j, mask1, mask2)
                updated_box_array.append(new_box)
                updated_conf_list.append(new_conf)
                updated_class_id.append(new_class)
                updated_mask_list.append((new_mask, new_tl_x, new_tl_y, new_w, new_h))
                combined_indices.update([i, j])
                break
        if not overlap:
            updated_box_array.append(box1)
            updated_conf_list.append(conf_list[i])
            updated_class_id.append(cls_id_list[i])
            updated_mask_list.append(mask1)
    updated_box_array = np.array(updated_box_array)
    return updated_box_array, updated_conf_list, updated_class_id, updated_mask_list