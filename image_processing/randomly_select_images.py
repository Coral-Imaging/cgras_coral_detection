#! /usr/bin/env python3

# quick script to grab a desired number of target random images from source to target directory
# read from a source folder, then grab a random assortment of files and copy those into a different target folder
# the intended behaviour is to glob all the photos in the sub-directories of the specified source folder, and then amalgamate them into a single folder containing X images

# NOTE images that are unsuitable for training can be manually removed. The number of subtracted images can then be determined manually and then editted
# and then use the append option, and iterate until the original intended target number of images is reached

import os
import shutil
import random
from pathlib import Path
import sys

def copy_images(imgs_list, target_dir):
    # copy images (list of path objects) to target_dir
    # assume directory exists
    for i, img_path in enumerate(imgs_list):
                print(f'img {i+1}/{len(imgs_list)}')
                shutil.copy(img_path, 
                            os.path.join(target_dir, img_path.name))

# source folder
source_dir = '/home/dtsai/Data/cgras_datasets/cgras_2024_aims_camera_trolley/Amag_Oct_2024/'

# target folder
target_dir = '/home/dtsai/Data/cgras_datasets/cgras_2024_aims_camera_trolley/for_labelling/'

# target number of images
target_images = 100

# check to make sure assume target images is greater than number of images in the folder
print(f'Gathering list of images in all sub-directories of source directory: {source_dir}')
img_list = sorted(Path(source_dir).rglob('CGRAS_Amag_*.jpg'))
n_img = len(img_list)

if target_images > len(img_list):
    print(f'Number of target images: {target_images}')
    print(f'Number of images in source directory (including all sub-directories): {len(img_list)}')
    print(f'ERROR: number of target images is greater than number of images in source directory')
    sys.exit(1)

# randomly sample
print(f'randomly sample {target_images} images out of {len(img_list)}')
print(f'Percent sampled from source_dir: {target_images / len(img_list) * 100}%')
imgs_rng = random.sample(img_list, target_images)

# take those image names and then move them into a new folder:
# be sure to clear folder ahead of time - or ask user
if not os.path.exists(target_dir):
    print(f'Target_dir does not yet exist, making new directory')
    os.makedirs(target_dir)
    copy_images(imgs_rng, target_dir)
else:
    # if directory exists, wait for user input
    i = 0
    print(f'Target_dir already exists.')
    while True and i < 10:
        user_input = input(' Do you want to (d) Delete and recreate the directory, (a) append to the directory---possibly overwriting files, or (e) exit the operation? ')
        if user_input == "d": 
            print('Removing existing folder, creating new target_dir, copying to new target_dir')
            shutil.rmtree(target_dir)
            os.makedirs(target_dir)
            copy_images(imgs_rng, target_dir)
            break
        elif user_input == "a":
            print('Copying over existing target_dir')
            copy_images(imgs_rng, target_dir)
            break
        elif user_input == "e":
            print('Exiting operation')
            sys.exit(1)

        else:
            print('Invalid user input. Please enter ''d'', ''a'', or ''e''')

        i+=1

    # yes: delete and overwrite
    # no: append
    # exit: stop

print('done')

# import code
# code.interact(local=dict(globals(), **locals()))