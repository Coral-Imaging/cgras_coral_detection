#! /usr/bin/env python3

"""resize_files.py

resize data files in specified folder to specified resolutions for training at different resolutions
Dorian Tsai
2024 July 26
"""
  #Probably legacy code
  
import os
import cv2 as cv
import glob
import shutil
  
# to handle train/val folder splits of image files
def get_rest_of_relative_path(file_path, base_folder):
    # Split the path into the base folder and the rest
    base, rest = os.path.split(file_path)
    while os.path.basename(base) != os.path.basename(base_folder):
        rest = os.path.join(os.path.basename(base), rest)
        base = os.path.dirname(base)
    return rest

def copy_labels_dir(label_list, out_dir, label_dir='train/labels'):
    # copy laels from glob list to directory
    for label_name in label_list:
        destination = os.path.join(out_dir, label_dir,os.path.basename(label_name))
        if not os.path.lexists(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(label_name, destination)
        
# specify data directory
# should be tiled images

data_dir = '/home/dorian/Data/cgras_data_copied_2240605/samples/cgras_data_copied_2240605_split_n_tilled'
print(f'data_dir = {data_dir}')

# specify resolution(s)/desired image sizes
image_sizes = [240]
print(f'image_sizes = {image_sizes}')

# loop over image sizes, resize and save into folder
for i in image_sizes:
    
    train_image_list = glob.glob(os.path.join(data_dir,'train/images/*.jpg'))
    val_image_list = glob.glob(os.path.join(data_dir,'valid/images/*.jpg'))
    image_list = train_image_list + val_image_list
    print(f'length of image_list = {len(image_list)}')
    
    train_label_list = glob.glob(os.path.join(data_dir,'train/labels/*.txt'))
    val_label_list = glob.glob(os.path.join(data_dir,'valid/labels/*.txt'))
    
    
    # get aspect ratio, assume all images in same dir have same aspect ratio
    if len(image_list) > 1:
        image = cv.imread(image_list[0])
        height, width, chan = image.shape
        ar = width / height
    else:
        ValueError('no images in image_list')
    
    # specify output data directory
    out_dir = os.path.join(data_dir,str(i)+'p')
    os.makedirs(out_dir, exist_ok=True)
    print(f'out_dir = {out_dir}')
    
    # copy labels over
    copy_labels_dir(train_label_list, out_dir, 'train/labels')
    copy_labels_dir(val_label_list, out_dir, 'valid/labels')
        
    # loop over all images in list
    for image_name in image_list:
        
        # open image (bgr)
        image = cv.imread(image_name)
        print(f'reading image = {image_name}')
        
        # resize image, assuming width is largest dimension
        width_r = i
        height_r = round(width_r / ar)
        image_r = cv.resize(image, (width_r, height_r), interpolation=cv.INTER_LINEAR)
        
        # save image (bgr)
        common_name = get_rest_of_relative_path(image_name, data_dir)
        # save_file = os.path.join(out_dir, common_name[:-4]+'_'+str(i)+'.jpg')
        save_file = os.path.join(out_dir, common_name)
        print(f'saving image = {save_file}')
        if not os.path.lexists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
        cv.imwrite(save_file, image_r)
    
print('done')
