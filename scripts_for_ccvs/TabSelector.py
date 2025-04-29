#!/usr/bin/env python3

# script to randomly select tabs (3x4 patches) across tile and date
# write them to spreadsheet for easy of keeping track
import random
import matplotlib.pyplot as plt

select_width = 4
select_height = 3 

tile_width = 20
tile_height = 20

# Randomly select starting points for width and height
width_selector = random.randint(0, tile_width - select_width)
height_selector = random.randint(0, tile_height - select_height)

# Generate the selected ranges
tab_selection_width = [w for w in range(width_selector, width_selector + select_width)]
tab_selection_height = [h for h in range(height_selector, height_selector + select_height)]

print(f'tab_selection_width: {tab_selection_width}')
print(f'tab_selection_height: {tab_selection_height}')

# Visualization
array = [['white' for _ in range(tile_width)] for _ in range(tile_height)]
for h in tab_selection_height:
    for w in tab_selection_width:
        array[h][w] = 'red'

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
plt.title('Tab Selection Visualization for Tile X, Date Y', fontsize=16)
for i, row in enumerate(array):
    for j, color in enumerate(row):
        ax.add_patch(plt.Rectangle((j, tile_height - i - 1), 1, 1, facecolor=color, edgecolor='black', linewidth=2))
        # Add index text inside the rectangle
        ax.text(j + 0.5, tile_height - i - 0.5, f'{j},{i}', color='black', ha='center', va='center', fontsize=6)

ax.set_xlim(0, tile_width)
ax.set_ylim(0, tile_height)
ax.set_aspect('equal')
ax.axis('off')  # Turn off the axes for a cleaner look

plt.savefig('tab_selection.png', dpi=300, bbox_inches='tight')
plt.show()