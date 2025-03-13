#! /usr/bin/env python3

""" predict_segmenter.py
run segmentation model on folder of images
"""

from ultralytics import YOLO
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
#Probably legacy code
# load model
model_path = '/home/dorian/Code/cgras_ws/cgras_settler_counter/segmenter/weights/20230606_overfit.pt'
# load custom model (YOLOv8)
model = YOLO(model_path)

# output directory (for plots/detections)
out_dir = '/home/dorian/Data/cgras_datasets/cgras_dataset_20230421/predict'
os.makedirs(out_dir, exist_ok=True)

# image directory
img_dir = '/home/dorian/Data/cgras_datasets/cgras_dataset_20230421/train/images'
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))



for i, img_name in enumerate(img_list):
    print(f'{i}/{len(img_list)}: {os.path.basename(img_name)}')
    # model inference
    results = model(img_name, 
                    save=True, 
                    save_txt=True, 
                    save_conf=True, 
                    boxes=True,
                    conf=0.3, # intentionally very low for debugging purposes
                    agnostic_nms=True)
    
    res_plotted = results[0].plot()
    res_rgb = cv.cvtColor(res_plotted, cv.COLOR_BGR2RGB)
    # save image
    img_save_name = os.path.basename(img_name).rsplit('.')[0] + '_box.jpg'
    plt.imsave(os.path.join(out_dir, img_save_name), res_rgb)
    # plt.show()
    