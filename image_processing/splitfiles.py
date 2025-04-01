#! /usr/bin/env python3

""" splitfiles.py
    This script is used to split a dataset into training, testing and validation sets.
"""

import os
import zipfile
import shutil
import glob
import random

train_ratio = 0.70
test_ratio = 0.15
valid_ratio = 0.15

data_locaton = '/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/20250318_improved_label_dataset_split_subimages/test'
save_dir_for_split = '/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/Reduced_dataset_patches/fixxed_labels' 

os.makedirs(save_dir_for_split, exist_ok=True)

imagelist = glob.glob(os.path.join(data_locaton+'/images', '*.jpg'))
txtlist = glob.glob(os.path.join(data_locaton+'/labels', '*.txt'))
txtlist.sort()
imagelist.sort()
imgno = len(txtlist) 
noleft = imgno
print(f"processing {len(imagelist)}")

def check_ratio(test_ratio,train_ratio,valid_ratio):
    if(test_ratio>1 or test_ratio<0): ValueError(test_ratio,f'test_ratio must be > 1 and test_ratio < 0, test_ratio={test_ratio}')
    if(train_ratio>1 or train_ratio<0): ValueError(train_ratio,f'train_ratio must be > 1 and train_ratio < 0, train_ratio={train_ratio}')
    if(valid_ratio>1 or valid_ratio<0): ValueError(valid_ratio,f'valid_ratio must be > 1 and valid_ratio < 0, valid_ratio={valid_ratio}')
    if not((train_ratio+test_ratio+valid_ratio)==1): ValueError("sum of train/val/test ratio must equal 1")
check_ratio(test_ratio,train_ratio,valid_ratio)

validimg, validtext, testimg, testtext = [], [], [], []

# function to seperate files into different lists randomly while retaining the same .txt and .jpg name in the specific type of list
def seperate_files(number,newimglist,newtxtlist,oldimglist,oldtxtlist):
    for i in range(int(number)):
        r = random.randint(0, len(oldtxtlist) - 1)
        newimglist.append(oldimglist[r])
        newtxtlist.append(oldtxtlist[r])
        oldimglist.remove(oldimglist[r])
        oldtxtlist.remove(oldtxtlist[r])
    return oldimglist, oldtxtlist

#pick some random files
imagelist, txtlist = seperate_files(imgno*valid_ratio,validimg,validtext,imagelist,txtlist)
imagelist, txtlist = seperate_files(imgno*test_ratio,testimg,testtext,imagelist,txtlist)
print(f"random files selected, {len(validimg)} validation images, {len(testimg)} testing images")

def copy_link(src, dst):
    """function to preserve symlinks of src file, otherwise default to copy"""
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, os.path.join(dst, os.path.basename(src)))
    else:
        shutil.copy(src, dst)

def clean_dirctory(savepath):
    """function to make sure the directory is empty"""
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath, exist_ok=True)

def move_file(filelist,savepath,second_path):
    """function to move a list of files, by cleaning the path and copying and preserving symlinks"""
    output_path = os.path.join(savepath, second_path)
    #clean_dirctory(output_path)
    os.makedirs(output_path, exist_ok=True)
    for i, item in enumerate(filelist):
        copy_link(item, output_path)

max_files = 16382

def split_and_move_files(file_list, image_list, save_dir, prefix, max_files):
    if len(file_list) >= max_files:
        print(f"{prefix} list exceeds max file number of: {max_files} at length: {len(file_list)}, splitting into multiple directories")
        split = len(file_list) // max_files
        for i in range(split):
            split_file_list = file_list[:max_files]
            split_image_list = image_list[:max_files]
            file_list = file_list[max_files:]
            image_list = image_list[max_files:]
            print(f"moving {len(split_file_list)} into {prefix}_{i}/labels")
            move_file(split_file_list, save_dir, f'{prefix}_{i}/labels')
            print(f"moving {len(split_image_list)} into {prefix}_{i}/images")
            move_file(split_image_list, save_dir, f'{prefix}_{i}/images')

        print(f"moving {len(file_list)} into {prefix}_{i+1}/labels")
        move_file(file_list, save_dir, f'{prefix}_{i+1}/labels')
        print(f"moving {len(image_list)} into {prefix}_{i+1}/images")
        move_file(image_list, save_dir, f'{prefix}_{i+1}/images')
    else:
        print(f"moving {len(file_list)} into {prefix}/labels")
        move_file(file_list, save_dir, f'{prefix}/labels')
        print(f"moving {len(image_list)} into {prefix}/images")
        move_file(image_list, save_dir, f'{prefix}/images')

split_and_move_files(txtlist, imagelist, save_dir_for_split, 'train', max_files)
split_and_move_files(validtext, validimg, save_dir_for_split, 'valid', max_files)
split_and_move_files(testtext, testimg, save_dir_for_split, 'test', max_files)

print("split complete")

import code
code.interact(local=dict(globals(), **locals()))

#alternative method
# print("Usinf SKlearn file split")

# from sklearn.model_selection import train_test_split 

# X_train, X_test, y_train, y_test = train_test_split(imagelist, txtlist, test_size=0.30, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

