#!/bin/bash
#Author: Reginald Wardleworth
#Date: 2024-09-05
#Purpose: This script will run the ROI_2_CSV.py script on all the ROIs in the zip files in the base_dir

base_dir="/home/wardlewo/Reggie/data/SCU_Pdae_Data/T23 (Done)/ROIs"

find "$base_dir" -type f -name "*.zip" | while read -r f
do
    echo "$f"
    python ROI_2_CSV.py "$f"
done
    