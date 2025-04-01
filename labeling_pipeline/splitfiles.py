#! /usr/bin/env python3

""" splitfiles.py
    This script is used to split a dataset into training, testing and validation sets.
"""

import os
import zipfile
import shutil
import glob
import random


class DatasetSplitter:
    def __init__(self, data_location, save_dir, train_ratio=0.70, test_ratio=0.15, valid_ratio=0.15, max_files=16382):
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.data_location = data_location
        self.save_dir = save_dir
        self.max_files = max_files
        
        # Make sure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Check ratios
        self.check_ratio(self.test_ratio, self.train_ratio, self.valid_ratio)
    
    def check_ratio(self, test_ratio, train_ratio, valid_ratio):
        if(test_ratio>1 or test_ratio<0): ValueError(test_ratio,f'test_ratio must be > 1 and test_ratio < 0, test_ratio={test_ratio}')
        if(train_ratio>1 or train_ratio<0): ValueError(train_ratio,f'train_ratio must be > 1 and train_ratio < 0, train_ratio={train_ratio}')
        if(valid_ratio>1 or valid_ratio<0): ValueError(valid_ratio,f'valid_ratio must be > 1 and valid_ratio < 0, valid_ratio={valid_ratio}')
        if not((train_ratio+test_ratio+valid_ratio)==1): ValueError("sum of train/val/test ratio must equal 1")
    
    def seperate_files(self, number, newimglist, newtxtlist, oldimglist, oldtxtlist):
        """function to seperate files into different lists randomly while retaining the same .txt and .jpg name in the specific type of list"""
        for i in range(int(number)):
            r = random.randint(0, len(oldtxtlist) - 1)
            newimglist.append(oldimglist[r])
            newtxtlist.append(oldtxtlist[r])
            oldimglist.remove(oldimglist[r])
            oldtxtlist.remove(oldtxtlist[r])
        return oldimglist, oldtxtlist
    
    def copy_link(self, src, dst):
        """function to preserve symlinks of src file, otherwise default to copy"""
        if os.path.islink(src):
            linkto = os.readlink(src)
            os.symlink(linkto, os.path.join(dst, os.path.basename(src)))
        else:
            shutil.copy(src, dst)
    
    def clean_dirctory(self, savepath):
        """function to make sure the directory is empty"""
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
        os.makedirs(savepath, exist_ok=True)
    
    def move_file(self, filelist, savepath, second_path):
        """function to move a list of files, by cleaning the path and copying and preserving symlinks"""
        output_path = os.path.join(savepath, second_path)
        os.makedirs(output_path, exist_ok=True)
        for i, item in enumerate(filelist):
            self.copy_link(item, output_path)
    
    def split_and_move_files(self, file_list, image_list, save_dir, prefix):
        if len(file_list) >= self.max_files:
            print(f"{prefix} list exceeds max file number of: {self.max_files} at length: {len(file_list)}, splitting into multiple directories")
            split = len(file_list) // self.max_files
            for i in range(split):
                split_file_list = file_list[:self.max_files]
                split_image_list = image_list[:self.max_files]
                file_list = file_list[self.max_files:]
                image_list = image_list[self.max_files:]
                print(f"moving {len(split_file_list)} into {prefix}_{i}/labels")
                self.move_file(split_file_list, save_dir, f'{prefix}_{i}/labels')
                print(f"moving {len(split_image_list)} into {prefix}_{i}/images")
                self.move_file(split_image_list, save_dir, f'{prefix}_{i}/images')
    
            print(f"moving {len(file_list)} into {prefix}_{i+1}/labels")
            self.move_file(file_list, save_dir, f'{prefix}_{i+1}/labels')
            print(f"moving {len(image_list)} into {prefix}_{i+1}/images")
            self.move_file(image_list, save_dir, f'{prefix}_{i+1}/images')
        else:
            print(f"moving {len(file_list)} into {prefix}/labels")
            self.move_file(file_list, save_dir, f'{prefix}/labels')
            print(f"moving {len(image_list)} into {prefix}/images")
            self.move_file(image_list, save_dir, f'{prefix}/images')
    
    def split_dataset(self):
        # Load image and text files
        imagelist = glob.glob(os.path.join(self.data_location+'/images', '*.jpg'))
        txtlist = glob.glob(os.path.join(self.data_location+'/labels', '*.txt'))
        txtlist.sort()
        imagelist.sort()
        imgno = len(txtlist)
        print(f"processing {len(imagelist)}")
        
        # Initialize empty lists for validation and test sets
        validimg, validtext, testimg, testtext = [], [], [], []
        
        # Separate files into validation, test, and train sets
        imagelist, txtlist = self.seperate_files(imgno*self.valid_ratio, validimg, validtext, imagelist, txtlist)
        imagelist, txtlist = self.seperate_files(imgno*self.test_ratio, testimg, testtext, imagelist, txtlist)
        print(f"random files selected, {len(validimg)} validation images, {len(testimg)} testing images")
        
        # Split and move files to their respective directories
        self.split_and_move_files(txtlist, imagelist, self.save_dir, 'train')
        self.split_and_move_files(validtext, validimg, self.save_dir, 'valid')
        self.split_and_move_files(testtext, testimg, self.save_dir, 'test')
        
        print("split complete")
        
        # Return the file lists for potential further processing
        return {
            'train_images': imagelist,
            'train_labels': txtlist,
            'valid_images': validimg,
            'valid_labels': validtext,
            'test_images': testimg,
            'test_labels': testtext
        }


def main():
    # Create and use the class with original parameters
    splitter = DatasetSplitter(
        data_location='/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/20250318_improved_label_dataset_split_subimages/test',
        save_dir='/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/Reduced_dataset_patches/fixxed_labels'
    )
    splitter.split_dataset()


if __name__ == "__main__":
    main()
    # Keep the interactive mode for debugging when run directly
    import code
    code.interact(local=dict(globals(), **locals()))

# Alternative method commented out for reference
# print("Usinf SKlearn file split")
# from sklearn.model_selection import train_test_split 
# X_train, X_test, y_train, y_test = train_test_split(imagelist, txtlist, test_size=0.30, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

