#! /usr/bin/env python3

""" extract images from cgras data samples into folder structure for detection"""

import os
import glob
import shutil

main_dir = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples'

folder1 = 'Week-2023-12-18'
subfolder1 = '231220-1248-1B-175'
subfolder2 = '231221-1238-1A-187'

imglist01 = sorted(glob.glob(os.path.join(main_dir+'/'+folder1+'/'+subfolder1, '*/*.jpg')))
imglist02 = sorted(glob.glob(os.path.join(main_dir+'/'+folder1+'/'+subfolder2, '*/*.jpg')))

folder2 = 'Week-2024-01-01'
folder3 = 'Week-2024-01-15'
folder4 = 'Week-2024-01-29'
folder5 = 'Week-2024-02-12'
folder6 = 'Week-2024-02-19'
folder7 = 'Week-2024-02-26'
folder8 = 'Week-2024-03-04' #is empty
folder9 = 'Week-2024-03-11'
folder10 = 'Week-2024-03-25' #is empty

imglist1 = sorted(glob.glob(os.path.join(main_dir+'/'+folder2, '*/*/*.jpg')))
imglist2 = sorted(glob.glob(os.path.join(main_dir+'/'+folder3, '*/*/*.jpg')))
imglist3 = sorted(glob.glob(os.path.join(main_dir+'/'+folder4, '*/*/*.jpg')))
imglist4 = sorted(glob.glob(os.path.join(main_dir+'/'+folder5, '*/*/*.jpg')))
imglist5 = sorted(glob.glob(os.path.join(main_dir+'/'+folder6, '*/*/*.jpg')))
imglist6 = sorted(glob.glob(os.path.join(main_dir+'/'+folder7, '*/*/*.jpg')))
imglist7 = sorted(glob.glob(os.path.join(main_dir+'/'+folder9, '*/*/*.jpg')))
imglist8 = sorted(glob.glob(os.path.join(main_dir+'/'+folder10, '*/*.jpg')))

imglist = imglist01 + imglist02 + imglist1 + imglist2 + imglist3 + imglist4 + imglist5 + imglist6 + imglist7 + imglist8

save_dir = os.path.join(main_dir, 'ultralytics_data')
os.makedirs(save_dir, exist_ok=True)

# for img in imglist:
#     shutil.copy(img, save_dir)

## manually remove unsuitable images

import code
code.interact(local=dict(globals(), **locals()))