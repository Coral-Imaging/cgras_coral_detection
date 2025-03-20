#! /usr/bin/env python3

""" cvatcoco_to_yolo.py
    This script is used to convert a COCO dataset to YOLO format.
    Works on masks and Bouding boxes.
"""

from ultralytics.data.converter import convert_coco
import os
import zipfile
import shutil
import glob
import random


#export from CVAT as COCO1.1
#comes in as ziped file with annotations/instances_default.json
export_name = '19-sep-coco_cgras2024.zip' #'cgras_coco.zip'
data_locaton = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cgras_data_copied_2240605_ultralytics_data' #location of /images folder
#data_locaton = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/tilling2/'
save_dir = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cvat_labels' #this a temp folder, labels will be save in data location/labels
download_dir = '/home/java/Downloads'

convert = False #true if want to convert coco to yolo
label_fix = False #true if coco has different order of classes
fill_in = True #if want to fill in blank text files

# Define a mapping for label transformations if needed
label_mapping = {
    '3': '4',
    '4': '5',
    '5': '6',
    '6': '7',
    '7': '8',
    '8': '9',
    '9': '10',
    '10': '3'
}

if convert:
    # convert coco to yolo
    with zipfile.ZipFile(os.path.join(download_dir, export_name), 'r') as zip_ref:
        zip_ref.extractall(download_dir)
            #annotation folder in downloads

    downloaded_labels_dir = os.path.join(download_dir, 'annotations')
    lable_dir = os.path.join(save_dir, 'labels')
    coco_labels = os.path.join(lable_dir, 'default')

    convert_coco(labels_dir=downloaded_labels_dir, save_dir=save_dir,
                    use_segments=True, use_keypoints=False, cls91to80=False)

    #move from subfolder in coco/labels/defult to where data is
    data_labels = os.path.join(data_locaton, 'labels')
    os.makedirs(data_labels, exist_ok=True)
    for filename in os.listdir(coco_labels):
        source = os.path.join(coco_labels, filename)
        destination = os.path.join(data_labels, filename)
        if os.path.isfile(source):
            shutil.move(source, destination)
    shutil.rmtree(save_dir) #tidy up

    print("conversion complete")

if label_fix:
    for i, label in enumerate(glob.glob(os.path.join(data_locaton, 'labels', '*.txt'))):
        print("processing label: ", label)
        with open(label, 'r') as file:
            data = file.readlines()
        with open(label, 'w') as file:
            for line in data:
                line = line.split()
                line[0] = label_mapping[line[0]]
                line[0] = str(int(line[0]))
                file.write(' '.join(line)+'\n')
    print("label fix complete")

#if images with no annotations
#need to add in a bank text file for labels
if fill_in:
    images_folder = os.path.join(data_locaton,'images')
    labels_folder = os.path.join(data_locaton,'labels')
    image_files = [os.path.splitext(file)[0] for file in os.listdir(images_folder)]
    label_files = [os.path.splitext(file)[0] for file in os.listdir(labels_folder)]
    missing_files = [file for file in image_files if file not in label_files]
    print(f'number of missing files: {len(missing_files)}')
    for file_name in missing_files:
        with open(os.path.join(labels_folder, file_name+'.txt'), 'w') as file:
            pass
    print ("blank text files added")


import code
code.interact(local=dict(globals(), **locals()))