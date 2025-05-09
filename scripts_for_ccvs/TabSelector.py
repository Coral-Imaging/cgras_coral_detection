#!/usr/bin/env python3

# script to randomly select tabs (3x4 patches) across tile and date
# write them to spreadsheet for easy of keeping track
import random
import matplotlib.pyplot as plt
from openpyxl import Workbook
import os
import glob
import yaml
import shutil
import re

class TabSelector:
    def __init__(self, yaml_data=None, select_width=4, select_height=3, tile_width=20, 
                 tile_height=20, output_dir=None, filename=None):
        """
        Initialize TabSelector with dimensions and output parameters
        
        Args:
            yaml_data (dict): YAML data containing tile configuration
            select_width (int): Width of the selection
            select_height (int): Height of the selection
            tile_width (int): Width of the tile
            tile_height (int): Height of the tile
            output_dir (str): Directory for output files
            filename (str): Base filename for output files (without extension)
        """
        self.yaml_data = yaml_data or {}
        self.select_width = select_width
        self.select_height = select_height
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.output_dir = output_dir or os.getcwd()
        self.filename = filename
        
        # Extract data from YAML if available
        self.extract_yaml_data()
        
        # Initialize variables that will be set by select_tabs
        self.tab_selection_width = None
        self.tab_selection_height = None
        self.array = None
    
    def extract_yaml_data(self):
        """
        Extract and process metadata from YAML data
        """
        # Set defaults
        self.tile_id = "T00"
        self.tile_name = "X"  
        self.date = "unknown_date"
        self.species = ""
        self.tank = ""
        self.week_number = ""
        
        if not self.yaml_data:
            return
            
        # Extract basic information directly from YAML
        self.tile_id = self.yaml_data.get('tile_id', self.tile_id)
        self.tile_name = self.yaml_data.get('tile_name', self.tile_id)
        
        # Extract additional information from batch time or other fields
        if 'batch_time' in self.yaml_data:
            batch_time = self.yaml_data['batch_time']
            # Extract date from batch_time format "YYYY-MM-DD HH:MM:SS"
            if isinstance(batch_time, str) and len(batch_time) >= 10:
                date_part = batch_time[:10].replace('-', '')
                if re.match(r'\d{8}', date_part):
                    self.date = date_part
        
        # Get species information
        self.species = self.yaml_data.get('species', "")
        
        # Extract information from image files if available
        if 'images' in self.yaml_data and self.yaml_data['images']:
            # Get the first image filename
            sample_filename = self.yaml_data['images'][0].get('file', '')
            
            # Extract the week part using regex
            week_match = re.search(r'_w(\d+)_', sample_filename)
            if week_match:
                self.week_number = week_match.group(1)  # Extract just the number part
                
            # Extract tank information
            tank_match = re.search(r'_([^_]+)_\d{8}_', sample_filename)
            if tank_match:
                self.tank = tank_match.group(1)  # Extract the tank identifier (e.g., MIS5a)
                
            # Extract date from filename if not set yet
            if self.date == "unknown_date":
                date_match = re.search(r'_(\d{8})_', sample_filename)
                if date_match:
                    self.date = date_match.group(1)
    
    def select_tabs(self):
        """
        Randomly select tabs within the tile dimensions
        """
        # Randomly select starting points for width and height
        width_selector = random.randint(0, self.tile_width - self.select_width)
        height_selector = random.randint(0, self.tile_height - self.select_height)
        
        # Generate the selected ranges
        self.tab_selection_width = [w for w in range(width_selector, width_selector + self.select_width)]
        self.tab_selection_height = [h for h in range(height_selector, height_selector + self.select_height)]
        
        print(f'tab_selection_width: {self.tab_selection_width}')
        print(f'tab_selection_height: {self.tab_selection_height}')
        
        # Create the selection array
        self.array = [['white' for _ in range(self.tile_width)] for _ in range(self.tile_height)]
        for h in self.tab_selection_height:
            for w in self.tab_selection_width:
                self.array[h][w] = 'red'
        
        return self.tab_selection_width, self.tab_selection_height
    
    def visualize_tabs(self, save_path=None):
        """
        Visualize the selected tabs
        
        Args:
            save_path (str): Path to save the visualization image
        """
        if self.array is None:
            self.select_tabs()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.title(f'Tab Selection Visualization for Tile {self.tile_name}, Date {self.date}', fontsize=16)
        
        for i, row in enumerate(self.array):
            for j, color in enumerate(row):
                ax.add_patch(plt.Rectangle(
                    (j, self.tile_height - i - 1), 1, 1, 
                    facecolor=color, edgecolor='black', linewidth=2
                ))
                # Add index text inside the rectangle
                ax.text(
                    j + 0.5, self.tile_height - i - 0.5, f'{j},{i}',
                    color='black', ha='center', va='center', fontsize=6
                )
        
        ax.set_xlim(0, self.tile_width)
        ax.set_ylim(0, self.tile_height)
        ax.set_aspect('equal')
        ax.axis('off')  # Turn off the axes for a cleaner look
        
        if save_path is None:
            if self.filename:
                # Use the original filename if available
                save_path = os.path.join(self.output_dir, f"{self.filename}.png")
            else:
                # Fall back to the old naming scheme if no filename provided
                save_path = os.path.join(self.output_dir, f'tab_selection_{self.tile_name}_{self.date}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {save_path}")
    
    def export_tabs(self, output_file=None):
        """
        Export tab coordinates to an Excel file
        
        Args:
            output_file (str): Path to save the Excel file
        """
        if self.tab_selection_width is None or self.tab_selection_height is None:
            self.select_tabs()
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Tab Selection"
        
        # Add headers
        ws.append(["Species", "Tank", "Tile", "Imaging Date", "Assessment time point (week)", 
                   "Width Index (column)", "Height Index (row)", "CGRAS Count", "Complete", 
                   "Manual Count", "Manual Tile Assessment Start", "Manual Tile Assessment End"])
        
        # Format date as YYYY-MM-DD if it's in YYYYMMDD format
        formatted_date = self.date
        if re.match(r'\d{8}', self.date):
            formatted_date = f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}"
        
        # Add tab coordinates
        for h in self.tab_selection_height:
            for w in self.tab_selection_width:
                # Create a row with values for specific columns
                row = [
                    "Amag",           # Species # TODO need to change CCVS to accept Amag not Acropora
                    self.tank,              # Tank
                    self.tile_id,           # Tile
                    formatted_date,         # Imaging Date
                    self.week_number,       # Assessment time point (week)
                    w,                      # Width Index (column)
                    h,                      # Height Index (row)
                    "",                     # CGRAS Count
                    "",                     # Complete
                    "",                     # Manual Count
                    "",                     # Manual Tile Assessment Start
                    ""                      # Manual Tile Assessment End
                ]
                ws.append(row)
        
        if output_file is None:
            if self.filename:
                # Use the original filename if available
                output_file = os.path.join(self.output_dir, f"{self.filename}.xlsx")
            else:
                # Fall back to the old naming scheme if no filename provided
                output_file = os.path.join(self.output_dir, f'tab_selection_{self.tile_name}_{self.date}.xlsx')
        
        wb.save(output_file)
        print(f"Tab coordinates saved to {output_file}")
        
        return output_file
    
    def run(self):
        """
        Main method to execute the tab selection workflow
        """
        print(f"Starting Tab Selection")
        print(f"Selection dimensions: {self.select_width}x{self.select_height}")
        print(f"Tile dimensions: {self.tile_width}x{self.tile_height}")
        
        self.select_tabs()
        self.visualize_tabs()
        self.export_tabs()
        
        print("Tab selection completed!")
    
    @staticmethod
    def process_yaml_directory(yaml_dir, output_dir=None, clean_output=True):
        """
        Process all YAML files in a directory and create tab selections for each
        
        Args:
            yaml_dir (str): Directory containing YAML files
            output_dir (str): Directory to save output files (defaults to yaml_dir)
            clean_output (bool): If True, clean out existing files in output directory
        """
        if output_dir is None:
            output_dir = yaml_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean out existing files if requested
        if clean_output:
            print(f"Cleaning output directory: {output_dir}")
            for file_pattern in ['*.png', '*.xlsx']:
                files = glob.glob(os.path.join(output_dir, file_pattern))
                for file in files:
                    try:
                        os.remove(file)
                        print(f"Removed: {file}")
                    except Exception as e:
                        print(f"Error removing {file}: {str(e)}")
        
        # Find all YAML files in the directory
        yaml_files = []
        for extension in ['*.yaml', '*.yml']:
            yaml_files.extend(glob.glob(os.path.join(yaml_dir, extension)))
        
        if not yaml_files:
            print(f"No YAML files found in {yaml_dir}")
            return
        
        print(f"Found {len(yaml_files)} YAML files to process")
        
        for yaml_file in yaml_files:
            try:
                print(f"\nProcessing {yaml_file}...")
                
                # Load YAML data
                with open(yaml_file, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                
                # Get base filename for output
                base_name = os.path.splitext(os.path.basename(yaml_file))[0]
                
                # Create a TabSelector instance with YAML data
                selector = TabSelector(
                    yaml_data=yaml_data,
                    select_width=yaml_data.get('select_width', 4),
                    select_height=yaml_data.get('select_height', 3),
                    tile_width=yaml_data.get('tile_width', 20),
                    tile_height=yaml_data.get('tile_height', 20),
                    output_dir=output_dir,
                    filename=base_name  # Pass the original filename
                )
                
                # Run the tab selection process
                selector.run()
                
            except Exception as e:
                print(f"Error processing {yaml_file}: {str(e)}")
        
        print(f"\nProcessed {len(yaml_files)} YAML files")


def main():
    import argparse
    import sys
    
    # Define default directories
    default_yaml_dir = '/home/dtsai/Code/cgras/cgras_settler_counter/scripts_for_ccvs/cgras_tile_config_files'
    default_output_dir = '/home/dtsai/Code/cgras/cgras_settler_counter/scripts_for_ccvs/output'
    
    # If no command line arguments are provided, process all YAML files in the default directory
    if len(sys.argv) == 1:
        print(f"No arguments provided. Processing all YAML files in: {default_yaml_dir}")
        print(f"Outputs will be saved to: {default_output_dir}")
        TabSelector.process_yaml_directory(default_yaml_dir, default_output_dir, clean_output=True)
        return
    
    # Otherwise, parse command line arguments for more specific control
    parser = argparse.ArgumentParser(description='Tab selector for CCVS')
    parser.add_argument('--yaml_dir', type=str, help='Directory containing YAML files to process')
    parser.add_argument('--output_dir', type=str, help='Directory to save output files')
    parser.add_argument('--clean', action='store_true', help='Clean output directory before processing')
    parser.add_argument('--select_width', type=int, default=4, help='Width of selection')
    parser.add_argument('--select_height', type=int, default=3, help='Height of selection')
    parser.add_argument('--tile_width', type=int, default=20, help='Width of tile')
    parser.add_argument('--tile_height', type=int, default=20, help='Height of tile')
    parser.add_argument('--tile_name', type=str, default='X', help='Name of tile')
    parser.add_argument('--date', type=str, default='Y', help='Date for visualization')
    
    args = parser.parse_args()
    
    if args.yaml_dir:
        # Process all YAML files in directory
        output_dir = args.output_dir or default_output_dir
        TabSelector.process_yaml_directory(args.yaml_dir, output_dir, clean_output=args.clean)
    else:
        # Create and run a single tab selector
        output_dir = args.output_dir or default_output_dir
        
        # Create output directory and clean if needed
        os.makedirs(output_dir, exist_ok=True)
        if args.clean:
            print(f"Cleaning output directory: {output_dir}")
            for file_pattern in ['*.png', '*.xlsx']:
                files = glob.glob(os.path.join(output_dir, file_pattern))
                for file in files:
                    try:
                        os.remove(file)
                    except Exception as e:
                        print(f"Error removing {file}: {str(e)}")
        
        # Create yaml data dictionary with command line parameters
        yaml_data = {
            'tile_id': args.tile_name,
            'tile_name': args.tile_name,
        }
        
        selector = TabSelector(
            yaml_data=yaml_data,
            select_width=args.select_width,
            select_height=args.select_height,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            output_dir=output_dir,
            filename=f"tab_selection_{args.tile_name}_{args.date}"
        )
        selector.date = args.date  # Override date from command line
        selector.run()


if __name__ == "__main__":
    main()