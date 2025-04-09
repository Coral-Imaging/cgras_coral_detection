#! /usr/bin/env/python3

""" run segment model on folder of images
save predicted bounding box results both as txt and .jpg
"""

from ultralytics import YOLO
import os
import glob
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from analysis.Utils import classes, class_colours

weights_file_path = '/home/java/Java/ultralytics/runs/segment/train4/weights/best.pt'
img_folder = '/home/java/Java/data/cgras_20230421/train/images'
save_dir = '/home/java/Java/data/cgras_20230421'


def save_image_predictions_bb(predictions, imgname, imgsavedir, class_colours, classes):
    """
    save predictions/detections (assuming predictions in yolo format) on image as bounding box
    """
    img = cv.imread(imgname)
    imgw, imgh = img.shape[1], img.shape[0]
    for p in predictions:
        x1, y1, x2, y2 = p[0:4].tolist()
        conf = p[4]
        cls = int(p[5])
        #extract back into cv lengths
        x1 = x1*imgw
        x2 = x2*imgw
        y1 = y1*imgh
        y2 = y2*imgh        
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), class_colours[classes[cls]], 4)
        cv.putText(img, f"{classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 1.5, class_colours[classes[cls]], 4)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, img)
    return True

def save_txt_predictions_bb(predictions, imgname, txtsavedir):
    """
    save predictions/detections [xn1, yn1, xn2, yn2, cls, conf]) as bounding box (x and y values normalised)
    """
    imgsavename = os.path.basename(imgname)
    txt_save_path = os.path.join(txtsavedir, imgsavename[:-4] + '_det.txt')
    with open(txt_save_path, "w") as file:
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])
            line = f"{x1} {y1} {x2} {y2} {cls} {conf}\n"
            file.write(line)

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)

# get predictions
print('Model Inference:')

imglist = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
imgsave_dir = os.path.join(save_dir, 'detections', 'detections_images')
txtsave_dir = os.path.join(save_dir, 'detections', 'detections_txt')
os.makedirs(imgsave_dir, exist_ok=True)
os.makedirs(txtsave_dir, exist_ok=True)
for i, imgname in enumerate(imglist):
    print(f'predictions on {i+1}/{len(imglist)}')
    if i >= 5: # for debugging purposes
        break
    results = model.predict(source=imgname, iou=0.5, agnostic_nms=True)
    boxes = results[0].boxes 
    pred = []
    for b in boxes:
        if torch.cuda.is_available():
            xyxyn = b.xyxyn[0]
            pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
    predictions = torch.tensor(pred)
    save_image_predictions_bb(predictions, imgname, imgsave_dir, class_colours, classes)
    save_txt_predictions_bb(predictions, imgname, txtsave_dir)
    import code
    code.interact(local=dict(globals(), **locals()))