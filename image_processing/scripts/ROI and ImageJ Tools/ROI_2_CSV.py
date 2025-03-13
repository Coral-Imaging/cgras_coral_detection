import roifile
import os 
import csv
import sys

#Author: Reginald Wardleworth
#Date: 2024-09-05
#Purpose: This script is used to convert ROI files to CSV format, whilst extracting the data the master CSV to take the class data.

def findCoral(roi):
    master = "/home/wardlewo/Reggie/data/SCU_Pdae_Data/MC.csv"
    with open(master, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == roi.name:
                return row[3]
        return "Not Found"

# Function to export merged ROI data to CSV
def export_to_csv(rois, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        # Write column headers
        writer.writerow(['name', 'type', 'top', 'left', 'bottom', 'right', 'n_coordinates', 'integer_coordinates','subpixel_coordinates'])
        # Iterate through ROI data and write to CSV
        for roi in rois:
            type = findCoral(roi)
            writer.writerow([roi.name, type, roi.top, roi.left, roi.bottom, roi.right, roi.n_coordinates, roi.integer_coordinates, roi.subpixel_coordinates])

def main(roi_directory):
    # Directory containing ROI files
    tifName = os.path.splitext(os.path.basename(roi_directory))[0]
    folderName = os.path.basename(os.path.dirname(os.path.dirname(roi_directory)))
    
    roi = roifile.roiread(roi_directory)
    
 
    if roi:
        # Export to CSV
        try:
            os.makedirs(f'/home/wardlewo/Reggie/data/{folderName}')
        except: 
            pass
        finally:    
            csv_path = f'/home/wardlewo/Reggie/data/{folderName}/{tifName}_ROI_Extract.csv'  
            export_to_csv(roi, csv_path)
            print(f'CSV file exported: {csv_path}')
    else:
        print('No ROI files found or error merging ROIs.')

if __name__ == '__main__':
    roi_directory = sys.argv[1]
    main(roi_directory)