#! /usr/bin/env python3

"""Code used or developed to do resolution test"""

import os
import glob
import shutil
import cv2 as cv


def copy_images_from_txt_list(import_dir_imgs, export_dir_imgs, txt_files_dri):
    """copy images from one folder to another if the name matches the txt files in the folder"""
    txt_files = sorted(glob.glob(os.path.join(txt_files_dri, '*.txt')))
    imgs = sorted(glob.glob(os.path.join(import_dir_imgs, '*.jpg')))
    wanted_names = [os.path.splitext(os.path.basename(txt))[0] for txt in txt_files]

    i = 0
    for img_path in imgs:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        if img_name in wanted_names:
            print(f"Copying {i}: {img_name} of {len(wanted_names)}")
            i+=1
            shutil.copy(img_path, export_dir_imgs)
    

def resize_images(image_dir, save_dir, new_size):
    """resize and save images in new location"""
    imgs = sorted(glob.glob(os.path.join(image_dir, '*/images/*.jpg')))
    for img_path in imgs:
        subfolder = img_path.split('/')[-3] 
        img_name = os.path.basename(img_path)
        subfolder_path = os.path.join(save_dir, subfolder, 'images')
        os.makedirs(subfolder_path, exist_ok=True)

        img = cv.imread(img_path)
        resized_img = cv.resize(img, (new_size, new_size))
        
        save_path = os.path.join(subfolder_path, img_name)
        cv.imwrite(save_path, resized_img)
    
    print(f"Copied {len(imgs)} images to {save_dir}")
        
def copy_txt(txt_import_dir, txt_export_dir):
    """copy txts from one top directory to another keeping train, test and validation split"""
    txts = sorted(glob.glob(os.path.join(txt_import_dir, '*/labels/*.txt')))
    for txt_path in txts:
        subfolder = txt_path.split('/')[-3] 
        txt_name = os.path.basename(txt_path)
        subfolder_path = os.path.join(txt_export_dir, subfolder, 'labels')
        os.makedirs(subfolder_path, exist_ok=True)
        
        shutil.copy(txt_path, subfolder_path)
    
    print(f"Copied {len(txts)} txts to {txt_export_dir}")

#TODO make full function 
def unsplit(import_dir, export_dir):
    """Assumed splitfiles run after this"""
    imgs = sorted(glob.glob(os.path.join(import_dir, '*/images/*.jpg')))
    txts = sorted(glob.glob(os.path.join(import_dir, '*/labels/*.txt')))
    for txt_path in txts:
        txt_name = os.path.basename(txt_path)
        subfolder_path = os.path.join(export_dir, 'labels')
        os.makedirs(subfolder_path, exist_ok=True)
        shutil.copy(txt_path, subfolder_path)
    print(f"Copied {len(txts)} txts to {export_dir}")
    for img_path in imgs:
        img_name = os.path.basename(img_path)
        subfolder_path = os.path.join(export_dir, 'images')
        os.makedirs(subfolder_path, exist_ok=True)
        shutil.copy(img_path, subfolder_path)
    print(f"Copied {len(imgs)} imgs to {export_dir}")


import_dir_imgs = "/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cgras20240826/valid/images"
txt_files_dri = "/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/resolution_test/640p/val/labels"
export_dir_imgs = "/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/resolution_test/640p/val/images"
#copy_images_from_txt_list(import_dir_imgs, export_dir_imgs, txt_files_dri)

# imgs_640p = "/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/resolution_test/640p/"
# new_size = 480
# save_dir = f"/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/resolution_test/{new_size}p/"
# os.makedirs(save_dir, exist_ok=True)
# resize_images(imgs_640p, save_dir, new_size)
# copy_txt(imgs_640p, save_dir)

import_dir = "/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/resolution_test/120p"
export_dir = "/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/resolution_test/120p_not_split"
unsplit(import_dir, export_dir)
