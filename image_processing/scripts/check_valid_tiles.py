import os

# label_dir = "/media/agoni/RRAP03/outputs/split_dataset/exported_2024_cgras_amag_T18_first10_100quality/train/labels"
label_dir = "/media/agoni/RRAP03/exported_labelled_from_cvat/exported_2024_cgras_amag_T01_first10_100quality/data/labels/Train"
for label_file in os.listdir(label_dir):
    path = os.path.join(label_dir, label_file)
    with open(path, "r") as f:
        lines = f.readlines()

    # if len(lines) == 0:
    #     print(f"⚠️ Empty label file: {label_file}")
    #     continue

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:  # A valid segmentation label should have at least 3 points (6 values)
            print(f"⚠️ Invalid segmentation format in {label_file}: {line}")
