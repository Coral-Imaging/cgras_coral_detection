#! /usr/bin/env python3

"""trian_segmenter.py
train basic yolov8 model for image segmentation
"""

from ultralytics import YOLO
import torch

data_file = '/mnt/hpccs01/home/wardlewo/Corals/cgras_settler_counter/segmenter/cgras_alive_dead_seg_20250205.yaml'
#data_file = sys.argv[1]

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO("yolov8n-seg.pt")
#model = YOLO("/home/java/Java/cslics/resolution_test_results/models/resolution_test_640/weights/Cslic_640_best.pt")
model.info()


#model.train(data=data_file, epochs=200, batch=10)

# classes arg is lightweight and simply ignore classes that are not included in the classes list, 
# during train, val and predict, it has no effect on model architecture.
model.train(data=data_file, 
            device      = 0,       #For HPC, set to 0 or delete otherwise      
            epochs      = 750, 
            batch       = 0.95,  
            project     = "LearningRate_20250319_cgras_segmentation_2022-2023_dataset_alive_dead",
            workers     = 12,
            patience    = 50,
            pretrained  = False,
            save        = True,
            save_period = 25,
            deterministic = False,
            imgsz       = 640,
            lr0         = 0.1,
            #Augmentation
            #HSV added via Albumentations
            scale       = 0.2,
            flipud      = 0.5,
            fliplr      = 0.5
            ) #test run

# # for interactive debugger in terminal:
# import code
# code.interact(local=dict(globals(), **locals()))