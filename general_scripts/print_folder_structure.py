#! /usr/bin/env python3

""" folder_structure.py
    This script generates a visual representation of the folder structure of a given directory.
    It can optionally include file names and ignore hidden files/folders.
    The output is printed to the console.
"""

import os

def print_directory_structure(startpath, indent_level=0, include_files=True, ignore_hidden=True):
    """ Recursively generates the directory structure as a string, optionally ignoring hidden files/folders. """
    structure = []
    
    for root, dirs, files in os.walk(startpath):
        # Ignore hidden directories
        if ignore_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]

        level = root.replace(startpath, "").count(os.sep)
        indent = "│   " * (level - indent_level) + "├── " if level > indent_level else ""
        structure.append(f"{indent}{os.path.basename(root)}/")

        if include_files:
            sub_indent = "│   " * (level - indent_level + 1) + "├── "
            for f in files:
                structure.append(f"{sub_indent}{f}")

    return "\n".join(structure)

if __name__ == "__main__":
    folder_path = input("Enter the folder path to scan: ").strip()
    include_files = input("Include file names? (yes/no): ").strip().lower() == "yes"
    ignore_hidden = input("Ignore hidden files/folders? (yes/no): ").strip().lower() == "yes"

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        structure = print_directory_structure(folder_path, include_files=include_files, ignore_hidden=ignore_hidden)
        print(structure)
    else:
        print("Invalid folder path. Please enter a valid directory.")