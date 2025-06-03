import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# === CONFIGURATION FLAGS ===
SHOW_RANGE = True           # Enables/disables shaded range (min/max)
INCLUDE_ZERO = True         # Adds a (0, 0) point to the curve
SET_LIMITS = True           # Forces xlim and ylim if enabled
CONVERT_TO_MINUTES = False  # Converts total_train_time from seconds to minutes
CONVERT_TO_HOURS = True     # Converts total_train_time from seconds to hours

# === DATA LOADING ===
csv_folder = '/media/java/RRAP03/data/outputs/experiments/num_image_exp/results'
all_csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

df_list = []
for file in all_csv_files:
    try:
        df = pd.read_csv(file, usecols=["train_size", "total_train_time"])
        df_list.append(df)
    except Exception as e:
        print(f"Skipping file {file} due to error: {e}")

if not df_list:
    raise ValueError("No valid CSV files with required columns were found.")

full_df = pd.concat(df_list, ignore_index=True)

# === UNIT CONVERSION ===
if CONVERT_TO_MINUTES:
    full_df['total_train_time'] = full_df['total_train_time'] / 60
elif CONVERT_TO_HOURS:
    full_df['total_train_time'] = full_df['total_train_time'] / 3600

# === PLOTTING ===
plt.figure(figsize=(12, 8))

grouped = full_df.groupby('train_size')['total_train_time']
mean_time = grouped.mean()
min_time = grouped.min()
max_time = grouped.max()

if INCLUDE_ZERO:
    mean_time.loc[0] = 0.0
    min_time.loc[0] = 0.0
    max_time.loc[0] = 0.0

# Ensure correct order for plotting
mean_time = mean_time.sort_index()
min_time = min_time.sort_index()
max_time = max_time.sort_index()

train_sizes = mean_time.index

if SHOW_RANGE:
    plt.fill_between(train_sizes, min_time, max_time, alpha=0.3, label='Training Time Range')

plt.plot(train_sizes, mean_time, label='Training Time Mean')

# === FINAL SETTINGS ===
plt.xlabel('Number of Training Images')

if CONVERT_TO_HOURS:
    ylabel = 'Total Training Time (hours)'
elif CONVERT_TO_MINUTES:
    ylabel = 'Total Training Time (minutes)'
else:
    ylabel = 'Total Training Time (seconds)'

plt.ylabel(ylabel)
# plt.title('Training Time vs Number of Training Images')

if SET_LIMITS:
    plt.xlim(0, 100)
    plt.ylim(bottom=0)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
