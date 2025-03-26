#! /usr/bin/env python3

"""
    Purpose of this is so that I could export the file names to excel to track the time it takes to process the images.
"""

import os
import pandas as pd
from datetime import datetime


def export_files_to_excel(folder_path, excel_path=None):
    """
    Export names of files in a folder to an Excel document.
    
    Args:
        folder_path (str): Path to the folder containing files
        excel_path (str, optional): Path where Excel file will be saved. 
                                   If None, saves in the current directory with timestamp.
    
    Returns:
        str: Path to the created Excel file
    """
    # Check if folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")
    
    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Create a DataFrame with file names
    df = pd.DataFrame(files, columns=['Filename'])
    
    # Create Excel path if not provided
    if excel_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"folder_files_{timestamp}.xlsx"
    
    # Export to Excel
    df.to_excel(excel_path, index=False)
    print(f"Successfully exported {len(files)} files to '{excel_path}'")
    
    return excel_path

# Example usage
if __name__ == "__main__":
    # Replace with your folder path
    folder_path = "/media/java/RRAP03/unlabelled/updated_2024_cgras_amag_TilesMix_first100_100quality"
    
    # Call the function
    export_files_to_excel(folder_path)