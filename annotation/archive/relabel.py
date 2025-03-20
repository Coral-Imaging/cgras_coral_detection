#! /usr/bin/env/python3

""" relabel yolo labels for certain classes
"""
import os
import glob


classes = ["recruit_live_white", "recruit_cluster_live_white", "recruit_symbiotic", "recruit_symbiotic_cluster", "recruit_partial",
           "recruit_cluster_partial", "recruit_dead", "recruit_cluster_dead", "grazer_snail", "pest_tubeworm", "unknown"]

#want resruit partial anr recruit cluster partial to be labeled as recruite live white and recruit cluster live white respectivly


#label dirs
label_dir1 = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cgras_data_copied_2240605_ultralytics_data/labels'
label_list1 = sorted(glob.glob(os.path.join(label_dir1, '*.txt')))
label_dir2 = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/cgras_data_copied_2240605_split_n_tilled'
label_list2 = sorted(glob.glob(os.path.join(label_dir2, '*/*/*.txt')))
label_dir3 = '/media/java/CGRAS-SSD/cgras_data_copied_2240605/samples/240805_split_n_tilled'
label_list3 = sorted(glob.glob(os.path.join(label_dir3, '*/*/*.txt')))

label_list = label_list1 + label_list2 +label_list3

for i, text_file in enumerate(label_list):
    print(f"Procesing {i} of {len(label_list)}")
    with open(text_file, 'r') as file:
        data = file.readlines()
    with open(text_file, 'w') as file:
        for line in data:
            line = line.split()
            if line[0] == '4': #recruit partial
                line[0] = '0' #recruit live white
            elif line[0] == '5': #recruit_cluster_partial
                line[0] = '1' # recruit_cluster_live_white
            line[0] = str(int(line[0]))
            file.write(' '.join(line)+'\n')

print("Labels updated")
import code
code.interact(local=dict(globals(), **locals()))