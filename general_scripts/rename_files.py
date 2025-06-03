import os
import re

def fix_week_number_and_room(directory):
    for filename in os.listdir(directory):
        # Match the existing renamed format with the incorrect room number
        match = re.match(r"(CGRAS_\w+)_MIS6b_([0-9]{8})_w[0-9]{1}_(T[0-9]{2}_[0-9]{2}\.jpg)", filename)
        if match:
            base_name, date, rest_of_filename = match.groups()
            
            # Construct the corrected filename with "MIS1a" and "w03"
            correct_filename = f"{base_name}_MIS1a_{date}_w3_{rest_of_filename}"

            # Rename only if incorrect
            if filename != correct_filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, correct_filename)
                os.rename(old_path, new_path)
                print(f"Fixed: {filename} -> {correct_filename}")

# Usage
directory = "/media/java/RRAP03/cgras_2024_aims_camera_trolley/corals_spawned_2024_nov/20241217_w3/Pdae_MIS1a"
fix_week_number_and_room(directory)
