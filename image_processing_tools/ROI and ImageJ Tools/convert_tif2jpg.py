#! /usr/bin/env python3

""" convert_tiff2jpg.py
"""

from PIL import Image
import os
split = True # if the tiff needs to be split

if split:
    Image.MAX_IMAGE_PIXELS = None

# tiff directory
tif_dir = '/media/wardlewo/cslics_ssd/SCU_Pdae_Data/RAWData/T23 (Done)'#'/home/dorian/Data/cgras_dataset_20230421/tif'

# jpg out directory
jpg_dir = '/media/wardlewo/cslics_ssd/SCU_Pdae_Data/RAWData/RAWImage/T23' #'/home/dorian/Data/cgras_dataset_20230421/jpg'
os.makedirs(jpg_dir, exist_ok=True)

for i, filename in enumerate(os.listdir(tif_dir)):
    if filename.endswith('.tif'):
        print(f'converting {i}: {filename}')
        if split:
            image = Image.open(os.path.join(tif_dir, filename))
            width, height = image.size
            segment_width = width // 3
            segment_height = height // 3
            for i in range(3):
                for j in range(3):
                    # Define the bounding box for the segment
                    left = j * segment_width
                    upper = i * segment_height
                    right = left + segment_width
                    lower = upper + segment_height
                    segment = image.crop((left, upper, right, lower))
                    # Save each segment as a JPG
                    segment_filename = f"{os.path.splitext(filename)[0]}_segment_{left}_{upper}_{right}_{lower}.jpg"
                    segment.convert('RGB').save(os.path.join(jpg_dir, segment_filename))
            image.close()
        else:
            # open TIFF image
            image = Image.open(os.path.join(tif_dir, filename))
            # convert TIFF to jpeg
            new_filename = filename.rsplit('.', 1)[0] + '.jpg'
            image.convert('RGB').save(os.path.join(jpg_dir, new_filename))

            image.close()

print('Conversion complete')
