#! /usr/bin/env/python3

"""
Visualise the predition results using a trained yolov8 weights file
    Works for both masks and bounding boxes, as well as options for SAHI
"""
#TODO: Rewrite this script to be class based
from ultralytics import YOLO
import os
import glob
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from Utils import classes, class_colours, overlap_boxes, combine_detections, callback
import supervision as sv

# Visualise options
ultralitics_version = False #set to true, if want example of ultralitics prediction
SAHI = False #set to true if using SAHI to look at 640p images
batch = True #image too big / too many predictions to do in one go
just_groundtruth = False #if ground truth without predictions wanted
bounding_boxes = False #if bounding boxes are to be visualised
batch_height, batch_width = 3000, 3000

# files / data locations
weights_file_path = "/mnt/hpccs01/home/wardlewo/20250205_cgras_segmentation_alive_dead/train7/weights/best.pt"
save_dir = '/mnt/hpccs01/home/wardlewo/Data/Visualisation'
img_folder = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/test_0/labels/images/'
txt_folder = '/mnt/hpccs01/home/wardlewo/Data/cgras/cgras_23_n_24_combined/20241219_improved_label_dataset_S+P+NegsReduced+Altered_Labels/test_0/labels/labels/'

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(weights_file_path).to(device)
slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640), overlap_ratio_wh=(0.1, 0.1))

# functions
def save_txt_predictions_masks(results, conf, class_list, save_path):
    """save_txt_predictions_masks
    writes a textfile for the segmentation results, saving the data as ultralytics does
    """
    masks = results[0].masks
    txt_results = []
    for i, r in enumerate(masks):
        txt_result1 = [int(class_list[i])]
        seg = masks[i].xyn[0].copy().reshape(-1)
        for j, s in enumerate(seg):
            txt_result1.append(seg[j])
        txt_result1.append(conf[i])
        txt_results.append(txt_result1)
    with open(save_path, 'w') as file:
        for txt_result in txt_results:
            for item in txt_result:
                file.write(str(item) + ' ')
            file.write('\n')

def add_ground_truth(image, txt, classes, class_colours, line_tickness, imgname):
    """add_ground_truth
    adds the ground truth annotiations onto an image. Assumed groundtruth is masks
    """
    line_tickness = 2
    height, width, _ = image.shape
    points_normalised, points, class_idx = [], [], []
    with open(txt, "r") as file:
        lines = file.readlines()
    for line in lines:
        data = line.strip().split()
        if len(data) < 1:
            print(f'Invalid data in {txt}: {line}')
            continue
        class_idx.append(int(data[0]))
        points_normalised.append([float(val) for val in data[1:]])
    for data in points_normalised:
        values = []
        try:
            for i in range(0, len(data), 2):
                x = round(data[i]*width)
                y = round(data[i+1]*height)
                values.extend([x,y])
            points.append(values)
        except:
            points.append(values)
            print(f'invalid line of data length: {len(data)}, related to img {imgname}')
    for idx in range(len(class_idx)):
        pointers = np.array(points[idx], np.int32).reshape(-1,2)
        cv.polylines(image, [pointers], True, (0, 0, 0), line_tickness)  # Make polylines black
        # Calculate the center right-most point
        text_x = max(pointers[:, 0])
        text_y = pointers[pointers[:, 0].argmax(), 1]
        cls_name = classes[class_idx[idx]]
        font_size = 1.5  # Reduced font size
        font_thickness = 2  # Reduced font thickness
        cv.putText(image, f"gt:{cls_name}", (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness)  # Make text black
    print(f'number of ground truth annotiations: {len(points)}')
    return image


def save_image_predictions_mask(results, image, imgname, save_path, conf, class_list, classes, class_colours, ground_truth=False, txt=None, negDataMine=False):
    """save_image_predictions_mask
    saves the predicted masks results onto an image, recoring confidence and class as well. 
    Can also show ground truth anotiations from the associated textfile (assumed annotiations are normalised xy corifinate points)
    """
    height, width, _ = image.shape
    masked = image.copy()
    line_tickness = int(round(width)/600)
    font_size = 1.5#int(round(line_tickness/2))
    font_thickness = 2#3*(abs(line_tickness-font_size))+font_size
    if results and results[0].masks:
        for j, m in enumerate(results.masks):
            xyn = np.array(m.xyn)
            xyn[0, :, 0] = (xyn[0, :, 0] * width)
            xyn[0, :, 1] = (xyn[0, :, 1] * height)
            points = xyn.reshape((-1, 1, 2)).astype(np.int32)
            cls_name = classes[int(class_list[j])]
            desired_color = class_colours[cls_name]
            if points is None or not points.any() or len(points) == 0:
                print(f'mask {j} encountered problem with points {points}, class is {cls_name}')
            else: 
                cv.fillPoly(masked, [points], desired_color)
                xmin = min(xyn[0, :, 0])
                ymin = min(xyn[0, :, 1])
                # Draw black outline
                cv.putText(image, f"{conf[j]:.2f}: {cls_name}", (int(xmin-20), int(ymin - 15)), cv.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness + 1)
                # Draw text in class color
                cv.putText(image, f"{conf[j]:.2f}: {cls_name}", (int(xmin-20), int(ymin - 15)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No masks found in {imgname}')
    
    if ground_truth & (txt is not None) & (txt != ''):     
        image = add_ground_truth(image, txt, classes, class_colours, line_tickness, imgname)

    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det_mask.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)

    # import code
    # code.interact(local=dict(globals(), **locals()))

def save_img_batch(image_cv, box_array, conf_list, cls_id_list, mask_list, image_name, save_dir):
    """save_img_batch
    saves the predicted masks from a batched image (no ground truth)
    """
    height, width, _ = image_cv.shape
    masked = image_cv.copy()
    line_tickness = int(round(width)/600)
    font_size = 2#int(round(line_tickness/2))
    font_thickness = 5#3*(abs(line_tickness-font_size))+font_size
    if conf_list is not None:
        for j, m in enumerate(mask_list):
            contours, _ = cv.findContours(m[0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
            for contour in contours:
                points = np.squeeze(contour)
                if len(points.shape) == 1:
                    points = points.reshape(-1, 1, 2)
                elif len(points.shape) == 2 and points.shape[1] != 2:
                    points = points.reshape(-1, 1, 2)
                points += np.array([m[1], m[2]]) #shift the points to the correct location
                cls = classes[int(cls_id_list[j])]
                desired_color = class_colours[cls]
                if points is None or not points.any() or len(points) == 0:
                    print(f'mask {j} encountered problem with points {points}, class is {cls}')
                else: 
                    cv.fillPoly(masked, [points], desired_color) 
        for t, b in enumerate(box_array):
            cls = classes[int(cls_id_list[t])]
            desired_color = class_colours[cls]
            # Draw black outline
            cv.putText(image_cv, f"{conf_list[t]:.2f}: {cls}", (int(b[0]-20), int(b[1] - 15)), cv.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness + 1)
            # Draw text in class color
            cv.putText(image_cv, f"{conf_list[t]:.2f}: {cls}", (int(b[0]-20), int(b[1] - 15)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No masks found in {image_name}')
    alpha = 0.5
    semi_transparent_mask = cv.addWeighted(image_cv, 1-alpha, masked, alpha, 0)
    imgsavename = os.path.basename(image_name)
    imgsave_path = os.path.join(save_dir, imgsavename[:-4] + '_det_mask.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)

def update_lists(box_list, conf_list, cls_id_list, mask_list, sliced_detections, x, y, x_end, y_end):
    #update the detection list for SAHI
    for box in sliced_detections.xyxy:
        box[0] += x
        box[1] += y
        box[2] += x
        box[3] += y
        box_list.append(box)
    for conf in sliced_detections.confidence:
        conf_list.append(conf)
    for cls_id in sliced_detections.class_id:
        cls_id_list.append(cls_id)
    for data in sliced_detections.data['class_name']:
        data_dict['class_name'].append(data)
    for mask in sliced_detections.mask:
        mask_resized = cv.resize(mask.astype(np.uint8), (x_end - x, y_end - y))
        rows, cols = np.where(mask_resized == 1)
        if len(rows) > 0 and len(cols) > 0:
            top_left_y = rows.min()
            bottom_right_y = rows.max()
            top_left_x = cols.min()
            bottom_right_x = cols.max()
            box_width = bottom_right_x - top_left_x + 1
            box_height = bottom_right_y - top_left_y + 1
            sub_mask = mask_resized[top_left_y:bottom_right_y + 1, top_left_x:bottom_right_x + 1]
            mask_list.append((sub_mask, top_left_x + x, top_left_y + y, box_width, box_height))  
    return box_list, conf_list, cls_id_list, mask_list

def save_img_sliced(slicer, image, image_file, save_dir):
    sliced_detections = slicer(image=image)
    masked = image.copy()
    for detection in sliced_detections:
        xyxy = detection[0].tolist()
        mask_array = detection[1].astype(np.uint8) 
        confidence = detection[2]
        class_id = detection[3]
        contours, _ = cv.findContours(mask_array, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = np.squeeze(contour)
            if len(points.shape) == 1:
                points = points.reshape(-1, 1, 2)
            elif len(points.shape) == 2 and points.shape[1] != 2:
                points = points.reshape(-1, 1, 2)
            cls_name = classes[class_id]
            desired_color = class_colours[cls_name]
            if points is None or not points.any() or len(points) == 0:
                print(f'mask encountered problem with points {points}, class is {cls_name}')
            else: 
                cv.fillPoly(masked, [points], desired_color) 
        cv.putText(image, f"{confidence:.2f}: {cls_name}", (int(xyxy[0]-20), int(xyxy[1] - 5)), cv.FONT_HERSHEY_SIMPLEX, 2, desired_color, 5)
    alpha = 0.30
    semi_transparent_mask = cv.addWeighted(image, 1-alpha, masked, alpha, 0)
    imgsave_path = os.path.join(save_dir, os.path.basename(image_file)[:-4] + '_t1.jpg')
    cv.imwrite(imgsave_path, semi_transparent_mask)
def save_image_predictions(results, image, imgname, save_path, classes, class_colours, ground_truth=False, txt=None):
    """save_image_predictions
    saves the predicted bb results onto an image, recoring confidence and class as well. 
    Can also show ground truth anotiations from the associated textfile (assumed annotiations are normalised xy corifinate points)
    """
    height, width, _ = image.shape
    line_tickness = int(round(width)/600)
    font_size = 1#int(round(line_tickness/2))
    font_thickness = 1#3*(abs(line_tickness-font_size))+font_size
    if results and results[0].boxes:
        for j, b in enumerate(results[0].boxes):
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_name = classes[int(b.cls_name.item())]
            desired_color = class_colours[cls_name]
            conf = b.conf.item()
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2),), desired_color, line_tickness*2)
            cv.putText(image, f"{conf:.2f}: {cls_name}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    else:
        print(f'No boxes found in {imgname}')
    if ground_truth & (txt is not None):
        with open(txt, "r") as file:
            lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            cls_name = classes[int(data[0])]
            desired_color = class_colours[cls_name]
            x1 = int(float(data[1])*width)
            y1 = int(float(data[2])*height)
            w = int(float(data[3])*width)
            h = int(float(data[4])*height)
            cv.rectangle(image, (x1-w, y1-h), (x1+h, y1+w), desired_color, line_tickness)
            cv.putText(image, f"{cls_name}", (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, font_size, desired_color, font_thickness)
    imgsavename = os.path.basename(imgname)
    imgsave_path = os.path.join(save_path, imgsavename[:-4] + '_det.jpg')
    cv.imwrite(imgsave_path, image)

    # run scipt
def main():
    print('Model Inference:')
    imglist = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
    txtlist = sorted(glob.glob(os.path.join(txt_folder, '*.txt')))
    imgsave_dir = save_dir
    os.makedirs(imgsave_dir, exist_ok=True)

    if not ultralitics_version and not SAHI and not just_groundtruth:
        for i, imgname in enumerate(imglist):
            print(f'predictions on {i+1}/{len(imglist)}')
            # if i >= 500: # for debugging purposes
            #     break 
            #     import code
            #     code.interact(local=dict(globals(), **locals()))
            
            image = cv.imread(imgname)
            results = model.predict(source=imgname, iou=0.5, agnostic_nms=True, imgsz=640)
            conf, class_list = [], [] 
            for j, b in enumerate(results[0].boxes):
                conf.append(b.conf.item())
                class_list.append(b.cls.item())
            
            txt = txtlist[i]
            ground_truth = True
            save_image_predictions_mask(results, image, imgname, imgsave_dir, conf, class_list, classes, class_colours, ground_truth, txt, False)

    if SAHI:
        for i, imgname in enumerate(imglist):
            print(f'SHAI predictions on {i+1}/{len(imglist)}')
            image = cv.imread(imgname)
            txt = txtlist[i]
            image_height, image_width = image.shape[:2]
            if batch:
                data_dict = {'class_name': []}
                box_list, conf_list, cls_id_list, mask_list = [], [], [], []
                print("Batching image")
                for y in range(0, image_height, batch_height):
                    for x in range(0, image_width, batch_width):
                        y_end = min(y + batch_height, image_height)
                        x_end = min(x + batch_width, image_width)
                        img= image[y:y_end, x:x_end]
                        sliced_detections = slicer(image=img)
                        if sliced_detections.confidence.size == 0:
                            print("No detections found in batch")
                            continue
                        box_list, conf_list, cls_id_list, mask_list = update_lists(box_list, conf_list, cls_id_list, mask_list, sliced_detections, x, y, x_end, y_end)
                conf_array = np.array(conf_list)
                box_array = np.array(box_list)
                updated_box_array, updated_conf_list, updated_class_id, updated_mask_list, overlap_count = combine_detections(box_array, conf_array, cls_id_list, mask_list)
                while overlap_count > 1:
                    conf_array = np.array(updated_conf_list)
                    box_array = np.array(updated_box_array)
                    print("update again")
                    updated_box_array, updated_conf_list, updated_class_id, updated_mask_list, overlap_count = combine_detections(box_array, conf_array, updated_class_id, updated_mask_list)
                image2 = add_ground_truth(image, txt, classes, class_colours, 10, imgname)
                save_img_batch(image, updated_box_array, updated_conf_list, updated_class_id, updated_mask_list, os.path.basename(imgname), save_dir)
            else:
                image = add_ground_truth(image, txt, classes, class_colours, 2)
                save_img_sliced(slicer, image, imgname, save_dir)

    if just_groundtruth:
        for i, imgname in enumerate(imglist):
            print(f'predictions on {i+1}/{len(imglist)}')
            image = cv.imread(imgname)
            txt = txtlist[i]
            ground_truth = True
            image = add_ground_truth(image, txt, classes, class_colours, 4, imgname)
            imgsave_path = os.path.join(save_dir, os.path.basename(imgname)[:-4] + '_det_mask.jpg')
            cv.imwrite(imgsave_path, image)
            # import code
            # code.interact(local=dict(globals(), **locals()))

    if ultralitics_version: #ultralytics code
        for i, imgname in enumerate(imglist):
            image = cv.imread(imgname)
            results = model.predict(source=image, iou=0.5, agnostic_nms=True)
            conf = []
            class_list = []
            #masks = results[0].masks
            for i, b in enumerate(results[0].boxes):
                conf.append(b.conf.item())
                class_list.append(b.cls.item())
                #seg = masks[i].xyn[0]
            for r in results:
                im_array = r.plot(conf=True, line_width=4, font_size=4, boxes=True)
                im = Image.fromarray(im_array[..., ::-1])
                im.show()
    if bounding_boxes:
        for i, imgname in enumerate(imglist):
            print(f'predictions on {i+1}/{len(imglist)}')
            #if i >= 10: # for debugging purposes
            #    break 
            #    import code
            #    code.interact(local=dict(globals(), **locals()))
            image = cv.imread(imgname)
            results = model.predict(source=imgname, iou=0.5, agnostic_nms=True, imgsz=640)    
            txt = txtlist[i]
            ground_truth = True
            save_image_predictions(results, image, imgname, imgsave_dir, classes, class_colours,  ground_truth, txt)
    print("DONE")
        




if __name__ == "__main__":
    main()

### OLD notes
#weight_file = "/home/java/Java/ultralytics/runs/segment/train9/weights/cgras_yolov8n-seg_640p_20231209.pt" #dorian used
#weights_file_path = '/media/wardlewo/cslics_ssd/SCU_Pdae_Data/split and tilling/ultralytics_output/train4/weights/best.pt' #trained on 640 imgsz dataset combined 22 and 23
#weights_file_path = "/home/java/Java/ultralytics/runs/segment/train21/weights/best.pt" #trained on 640 imgsz dataset combined 22 and 23