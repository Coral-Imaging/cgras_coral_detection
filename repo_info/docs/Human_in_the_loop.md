# Human in the Loop Labeling Pipeline

The following pipeline is designed to automate and streamline the Human-in-the-Loop labeling process. It consists of two main parts:

## 1. Labeling images
**Prerequisite:** *Assuming the image data is saved in a folder.*
1. Manually upload the images to CVAT.
2. Download the empty annotation file, in the CVAT1.1 format. 
    This file is refered to as a basefile, and example can be seen as [annotations.xml](https://github.com/Coral-Imaging/cgras_settler_counter/blob/main/annotations.xml)
    Extraction from zip may be needed.
3. Run the `annotation/roboflow_sahi.py` script, updating the file locations as needed, this will slice an image up, detect corals as segmentated masks, and save the detections in a zipped *'completed.xml'* file in CVAT1.1 format as masks. 
Alternativly;
3. Run the `annotation/predict_to_cvat.py` script with the correct data_location and basefile locations. The script will output a zipped *'completed.xml'* file. Upload this file to CVAT (using CVAT1.1 format). NOTE: the current script outputs predictions into CVAT as polyshapes, but this can easily be changed to masks by setting the output_as_masks to True.


## 2. Exporting annotated CVAT data and training a model
**Prerequisite:** *Assuming the annotated data exits on CVAT as polygon shapes.* 
    If data is as a mask in cvat, then the `poly_to_mask.py` script can be used to convert to polygon shapes (export as cvat 1.1 and import as cvat1.1)
1. Export the job/task dataset in the COCO1.0 format.
2. To create the yolo labels run the `cvatcoco_to_yolo.py` script. Making sure to update the file location and names as nessary. Also it can fill in any blank annotation files (just read the note before use).
3. The `splitfiles.py` script can split the data.
    If the data needs to be sliced for training with sahi or at high resolutions, use `tiling_images.py`
4. Manually create a `.yml` file specifing these locations, [cgras_20230421.yml](https://github.com/Coral-Imaging/cgras_settler_counter/blob/main/segmenter/cgras_20230421.yaml) is an example of this.
5. Run `train_segmenter.py` script making sure to update the yml file.

