import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# === CONFIGURATION FLAGS ===
SHOW_RANGE = True        # Set to False to disable shaded range area
INCLUDE_ZERO = False      # Set to False to exclude the (0, 0) data point
SET_LIMITS = False        # Set to False to let matplotlib auto-scale axes

# === DATA LOADING ===
csv_folder = '/media/java/RRAP03/data/outputs/experiments/num_image_exp/results'
all_csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

df_list = []
for file in all_csv_files:
    try:
        df = pd.read_csv(file, usecols=["train_size", "conf", "map50"])
        df_list.append(df)
    except Exception as e:
        print(f"Skipping file {file} due to error: {e}")

if not df_list:
    raise ValueError("No valid CSV files with required columns were found.")

full_df = pd.concat(df_list, ignore_index=True)

# === PLOTTING ===
plt.figure(figsize=(12, 8))

for conf_level in sorted(full_df['conf'].unique()):
    conf_df = full_df[full_df['conf'] == conf_level]

    grouped = conf_df.groupby('train_size')['map50']
    mean_map = grouped.mean()
    min_map = grouped.min()
    max_map = grouped.max()

    if INCLUDE_ZERO:
        mean_map.loc[0] = 0.0
        min_map.loc[0] = 0.0
        max_map.loc[0] = 0.0

    # Ensure correct order for plotting
    mean_map = mean_map.sort_index()
    min_map = min_map.sort_index()
    max_map = max_map.sort_index()

    train_sizes = mean_map.index

    if SHOW_RANGE:
        plt.fill_between(train_sizes, min_map, max_map, alpha=0.3, label=f'Conf {conf_level} range')

    plt.plot(train_sizes, mean_map, label=f'Conf {conf_level} mean')

# === FINAL SETTINGS ===
plt.xlabel('Number of Training Images')
plt.ylabel('mAP50')
plt.title('mAP50 Performance vs Training Image Count')

if SET_LIMITS:
    plt.xlim(0, 100)
    plt.ylim(0, 1)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
