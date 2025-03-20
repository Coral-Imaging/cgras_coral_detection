#!/bin/bash

# run the predict_to_cvat for each folder in daa/cgrass_2023 file

base_file="/home/java/Java/Cgras/cgras_settler_counter/annotations.xml"
base_img_location="/home/java/Java/data/cgras_20230421/train/images"
output_filename="/home/java/Downloads/complete.xml"
python_script="/home/java/Java/Cgras/cgras_settler_counter/annotation/predict_to_cvat.py"
DIR="/home/java/Java/data/somefolder"
echo 'Running counts'
for remote in $(cat $data_file); do
    echo "Processing line: $DIR/$remote"
    python3 "$python_script" "$base_file" "$DIR/$remote" "$output_filename"
done