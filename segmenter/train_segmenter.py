#! /usr/bin/env python3

"""trian_segmenter.py
train basic yolov8 model for image segmentation
"""

from ultralytics import YOLO
import torch

data_file = '/mnt/hpccs01/home/wardlewo/Data/cgras/Cgras_2023_dataset_labels_updated/Reduced_dataset_patches/cgras_2023+2024_dataset_seg_20250326.yaml'
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
            device      = [0,1,2,3],       #For HPC, set to 0 or delete otherwise      
            epochs      = 750, 
            batch       = 128,  
            project     = "runs/20250326_cgras_seg_2023-2024_dataset",
            name        = "20250326_8n_train_multiGpu_B128",
            workers     = 8,
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