#! /usr/bin/env python3

"""trian_segmenter.py
train basic yolov8 model for image segmentation
"""

from ultralytics import YOLO
import torch

data_file = '/mnt/hpccs01/home/wardlewo/Data/cgras/seg_test/2_class_alive_dead_corals/2240605_cgras/cgras_classifier.yaml'
#data_file = sys.argv[1]

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO("yolov8x-seg.pt")
#model = YOLO("/home/java/Java/cslics/resolution_test_results/models/resolution_test_640/weights/Cslic_640_best.pt")
model.info()


#model.train(data=data_file, epochs=200, batch=10)

# classes arg is lightweight and simply ignore classes that are not included in the classes list, 
# during train, val and predict, it has no effect on model architecture.
model.train(data=data_file, 
            device      = [0,1],            
            epochs      = 500, 
            batch       = 16,  
            project     = "cgras_segmentation_alive+dead_coral",
            workers     = 4,
            patience    = 50,
            pretrained  = False,
            save        = True,
            save_period = 25,
            deterministic = False,
            imgsz       = 640,
            #Augmentation
            #HSV added via Albumentations
            scale       = 0.2,
            flipud      = 0.5,
            fliplr      = 0.5
            ) #test run

# # for interactive debugger in terminal:
# import code
# code.interact(local=dict(globals(), **locals()))