#! /usr/bin/env python3

""" image_browser.py
    A simple image browser for viewing images in a directory.
    The user can select a species, tile, and image index to view images by week.
    The images are displayed in a canvas with zoom and pan functionality.
    Grid lines can be toggled on/off to show a 640x640 grid.
"""

import os
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import re
import numpy as np
import time

# Define base directory
BASE_DIR = "/media/agoni/RRAP03/cgras_2024_aims_camera_trolley"

# Options
SPECIES_OPTIONS = ["Amag", "Maeq", "Pdae", "Amil"]

class ImageBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Browser")
        
        # Configure grid to make the image area expandable
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(1, weight=1)  # Make the image row expandable

        # Control Frame for dropdowns
        control_frame = ttk.Frame(root)
        control_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)

        # Dropdowns for species, tile, and photo index
        self.species_var = tk.StringVar()
        self.species_dropdown = ttk.Combobox(control_frame, textvariable=self.species_var, values=SPECIES_OPTIONS, state="readonly")
        self.species_dropdown.set(SPECIES_OPTIONS[0])
        self.species_dropdown.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.species_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_tiles())

        self.tile_var = tk.StringVar()
        self.tile_dropdown = ttk.Combobox(control_frame, textvariable=self.tile_var, state="readonly")
        self.tile_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.tile_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_indices())

        self.image_var = tk.StringVar()
        self.image_dropdown = ttk.Combobox(control_frame, textvariable=self.image_var, state="readonly")
        self.image_dropdown.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self.image_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_weeks())

        # Create a canvas for the image with scrolling capability
        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Zoom and pan variables
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.start_x = 0
        self.start_y = 0
        self.is_panning = False
        
        # Image management
        self.full_size_image = None       # Original loaded image (PIL)
        self.original_dimensions = (0, 0)  # Original image dimensions
        self.display_image_obj = None     # Current display PhotoImage object
        self.image_id = None              # Canvas image ID
        self.grid_lines = []              # Grid line IDs
        self.last_update_time = 0         # For throttling updates
        self.update_pending = False       # Flag for pending updates
        self.update_scheduled = False     # Flag for scheduled updates
        
        # Bottom control frame
        bottom_frame = ttk.Frame(root)
        bottom_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.columnconfigure(2, weight=1)
        bottom_frame.columnconfigure(3, weight=1)

        # Add grid lines checkbox
        self.show_grid_var = tk.BooleanVar(value=False)
        self.grid_checkbox = ttk.Checkbutton(
            bottom_frame, 
            text="Show 640x640 Grid", 
            variable=self.show_grid_var,
            command=self.toggle_grid
        )
        self.grid_checkbox.grid(row=0, column=0, padx=5, pady=5)

        # Buttons for navigating weeks
        self.prev_button = ttk.Button(bottom_frame, text="Previous Week", command=self.prev_week)
        self.prev_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Reset zoom button
        self.reset_zoom_button = ttk.Button(bottom_frame, text="Reset Zoom", command=self.reset_zoom)
        self.reset_zoom_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.next_button = ttk.Button(bottom_frame, text="Next Week", command=self.next_week)
        self.next_button.grid(row=0, column=3, padx=5, pady=5)

        # Initialize variables
        self.image_list = []
        self.week_list = []
        self.current_week_index = 0
        self.current_image_path = None  # Store the current image path
        self._resize_job = None  # For resize event debouncing
        
        # Bind events
        self.root.bind("<Configure>", self.on_window_resize)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows and MacOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        
        # Set initial window size
        self.root.geometry("800x600")
        
        # Update the UI
        self.update_tiles()

    def toggle_grid(self):
        """Handle showing/hiding the grid lines"""
        if self.show_grid_var.get():
            self.draw_grid_lines()
        else:
            self.clear_grid_lines()

    def clear_grid_lines(self):
        """Remove all grid lines from the canvas"""
        for line_id in self.grid_lines:
            self.canvas.delete(line_id)
        self.grid_lines = []

    def draw_grid_lines(self):
        """Draw grid lines on the canvas over the image"""
        if not self.image_id:
            return
            
        # Clear existing grid lines
        self.clear_grid_lines()
        
        # Get image position and dimensions
        img_bbox = self.canvas.bbox(self.image_id)
        if not img_bbox:
            return
            
        img_x, img_y, img_x2, img_y2 = img_bbox
        img_width = img_x2 - img_x
        img_height = img_y2 - img_y
        
        # Calculate grid spacing
        orig_width, orig_height = self.original_dimensions
        scale_x = img_width / orig_width
        scale_y = img_height / orig_height
        
        grid_size_x = 640 * scale_x
        grid_size_y = 640 * scale_y
        
        # Draw vertical lines
        x = img_x
        while x <= img_x2:
            line_id = self.canvas.create_line(
                x, img_y, x, img_y2, 
                fill="red", width=1, tags="grid"
            )
            self.grid_lines.append(line_id)
            x += grid_size_x
            
        # Draw horizontal lines
        y = img_y
        while y <= img_y2:
            line_id = self.canvas.create_line(
                img_x, y, img_x2, y, 
                fill="red", width=1, tags="grid"
            )
            self.grid_lines.append(line_id)
            y += grid_size_y

    def on_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming."""
        # Determine the mouse position relative to the canvas
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Determine zoom direction and amount
        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.1  # Zoom in
        else:
            zoom_factor = 0.9  # Zoom out
        
        # Apply zoom
        self.zoom_to_point(x, y, zoom_factor)
        
    def zoom_to_point(self, x, y, factor):
        """Zoom in/out centered on a specific point."""
        if not self.full_size_image:
            return
            
        # Limit zoom level between 0.1 and 10
        new_zoom = self.zoom_level * factor
        if 0.1 <= new_zoom <= 10:
            # Calculate new pan position to keep point under cursor
            self.pan_x = x - (x - self.pan_x) * factor
            self.pan_y = y - (y - self.pan_y) * factor
            self.zoom_level = new_zoom
            
            # Throttle updates to improve performance
            self.schedule_update()
    
    def schedule_update(self):
        """Schedule an update to avoid too frequent redraws"""
        if not self.update_scheduled:
            self.update_scheduled = True
            self.root.after(50, self.process_scheduled_update)
    
    def process_scheduled_update(self):
        """Process any pending updates"""
        self.update_scheduled = False
        self.update_image_display()
    
    def on_mouse_down(self, event):
        """Handle mouse button press for panning."""
        self.start_x = event.x
        self.start_y = event.y
        self.is_panning = True
        self.canvas.config(cursor="fleur")  # Change cursor to indicate panning
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        self.is_panning = False
        self.canvas.config(cursor="")  # Reset cursor
    
    def on_mouse_move(self, event):
        """Handle mouse movement for panning when zoomed in."""
        if self.is_panning and self.zoom_level > 1.0:
            # Calculate the amount to pan
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            
            # Update pan position
            self.pan_x += dx
            self.pan_y += dy
            
            # Update start position for next movement
            self.start_x = event.x
            self.start_y = event.y
            
            # Throttle updates during panning
            self.schedule_update()
    
    def reset_zoom(self):
        """Reset zoom and pan to default values."""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_image_display()

    def on_window_resize(self, event):
        """Handle window resize events."""
        # Only respond to window resize, not internal widget configure events
        if event.widget == self.root:
            # Avoid excessive resizing by adding a small delay
            if hasattr(self, "_resize_job") and self._resize_job:
                self.root.after_cancel(self._resize_job)
            self._resize_job = self.root.after(100, self.reload_image)

    def reload_image(self):
        """Reload and resize the current image."""
        if self.current_image_path and os.path.exists(self.current_image_path):
            # Reset zoom when window is resized
            self.zoom_level = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.display_image(force_reload=True)

    def create_display_image(self):
        """Create a new display image based on current zoom level"""
        if not self.full_size_image:
            return None
            
        # Get available canvas size
        canvas_width = max(self.canvas.winfo_width(), 300)
        canvas_height = max(self.canvas.winfo_height(), 300)
        
        # Get original image size
        orig_width, orig_height = self.full_size_image.size
        
        # Calculate the base scale to fit the image in the canvas
        base_scale = min(canvas_width / orig_width, canvas_height / orig_height)
        
        # Apply zoom to the base scale
        scale = base_scale * self.zoom_level
        
        # Calculate new dimensions
        width = int(orig_width * scale)
        height = int(orig_height * scale)
        
        # Resize the image
        # Use NEAREST for fast zooming during interactive operations
        resample_method = Image.NEAREST if self.is_panning else Image.LANCZOS
        resized_img = self.full_size_image.resize((width, height), resample_method)
        
        # Convert to PhotoImage and return
        return ImageTk.PhotoImage(resized_img)

    def update_image_display(self):
        """Update the image display with current zoom and pan settings."""
        if not self.full_size_image:
            return
            
        # Create new display image
        self.display_image_obj = self.create_display_image()
        if not self.display_image_obj:
            return
            
        # Get size of the new image
        width = self.display_image_obj.width()
        height = self.display_image_obj.height()
        
        # Update canvas
        if self.image_id:
            # Update the existing image
            self.canvas.itemconfig(self.image_id, image=self.display_image_obj)
            # Center the image, considering pan
            x_pos = (self.canvas.winfo_width() - width) // 2 + self.pan_x
            y_pos = (self.canvas.winfo_height() - height) // 2 + self.pan_y
            self.canvas.coords(self.image_id, x_pos, y_pos)
            
            # Update grid lines if they're enabled
            if self.show_grid_var.get():
                self.draw_grid_lines()
        else:
            # Create new image on canvas
            x_pos = (self.canvas.winfo_width() - width) // 2 + self.pan_x
            y_pos = (self.canvas.winfo_height() - height) // 2 + self.pan_y
            self.image_id = self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.display_image_obj)
            
            # Add grid lines if enabled
            if self.show_grid_var.get():
                self.draw_grid_lines()

    def get_image_paths(self):
        """Finds all images matching the selected species, tile, and image index, sorted by week."""
        selected_species = self.species_var.get()
        selected_tile = self.tile_var.get()
        selected_index = self.image_var.get()

        self.image_list = []
        self.week_list = []

        for root, _, files in os.walk(BASE_DIR):
            for file in files:
                match = re.match(rf"CGRAS_{selected_species}_.+_(\d{{8}})_w(\d+)_({selected_tile})_({selected_index})\.jpg", file)
                if match:
                    week_number = int(match.group(2))
                    self.image_list.append((week_number, os.path.join(root, file)))
                    if week_number not in self.week_list:
                        self.week_list.append(week_number)

        self.image_list.sort(key=lambda x: x[0])  # Sort by week
        self.week_list.sort()
        self.current_week_index = 0 if self.week_list else -1  # Reset week index
        self.display_image()

    def update_tiles(self):
        """Updates available tile numbers for the selected species."""
        selected_species = self.species_var.get()
        tile_numbers = set()

        for root, _, files in os.walk(BASE_DIR):
            for file in files:
                match = re.match(rf"CGRAS_{selected_species}_.+_(\d{{8}})_w\d+_T(\d+)_\d+\.jpg", file)
                if match:
                    tile_numbers.add(f"T{int(match.group(2)):02d}")

        self.tile_dropdown["values"] = sorted(tile_numbers)
        if tile_numbers:
            self.tile_var.set(sorted(tile_numbers)[0])

        self.update_indices()

    def update_indices(self):
        """Updates available image indices for the selected species and tile."""
        selected_species = self.species_var.get()
        selected_tile = self.tile_var.get()
        image_numbers = set()

        for root, _, files in os.walk(BASE_DIR):
            for file in files:
                match = re.match(rf"CGRAS_{selected_species}_.+_(\d{{8}})_w\d+_{selected_tile}_(\d+)\.jpg", file)
                if match:
                    image_numbers.add(match.group(2))

        self.image_dropdown["values"] = sorted(image_numbers)
        if image_numbers:
            self.image_var.set(sorted(image_numbers)[0])

        self.update_weeks()

    def update_weeks(self):
        """Finds available weeks for the selected species, tile, and image index."""
        self.get_image_paths()

    def display_image(self, force_reload=False):
        """Displays the current image based on selected week."""
        if not self.image_list or self.current_week_index < 0:
            return
            
        week_number = self.week_list[self.current_week_index]
        img_path = next((path for wk, path in self.image_list if wk == week_number), None)

        if not img_path:
            return
            
        # If we're force reloading or this is a new image
        if force_reload or self.current_image_path != img_path:
            self.current_image_path = img_path
            
            # Clear any existing grid lines
            self.clear_grid_lines()
            
            # Reset zoom and pan when loading a new image
            if self.current_image_path != img_path:
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0
            
            # Load the image - process with OpenCV first for any needed transformations
            cv_img = cv2.imread(img_path)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # Store original dimensions
            self.original_dimensions = (cv_img.shape[1], cv_img.shape[0])  # width, height
            
            # Convert to PIL Image and store the full-size image
            self.full_size_image = Image.fromarray(cv_img)
            
            # Update the display
            self.update_image_display()
            
            self.root.title(f"Week {week_number} - {os.path.basename(img_path)}")

    def prev_week(self):
        """Show previous week's image."""
        if self.current_week_index > 0:
            self.current_week_index -= 1
            self.display_image()

    def next_week(self):
        """Show next week's image."""
        if self.current_week_index < len(self.week_list) - 1:
            self.current_week_index += 1
            self.display_image()

# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageBrowser(root)
    root.mainloop()