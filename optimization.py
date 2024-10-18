import h5py
import numpy as np
import pandas as pd
import os
import cc3d
from scipy.ndimage import binary_erosion
from itertools import product

# Directories
directoryInString = "2023-07-27_152203/thresholdedData/"
outputDirectoryInString = "2023-07-27_152203/segmentedVoxels/"
centroidsOutputDirectory = os.path.join(outputDirectoryInString, "centroids/")
os.makedirs(centroidsOutputDirectory, exist_ok=True)

def calculate_average_mass(j, structure):
    data_list = []

    for file in os.listdir(directoryInString):
        filename = os.fsdecode(file)
        if (int(filename) // 501) == j:
            with h5py.File(os.path.join(directoryInString, filename), 'r') as f:
                data = f['thresholdData'][:]
                if data.ndim == 2:
                    data = data[:, :, np.newaxis]
                data_list.append(data)

    if not data_list:
        return None

    mainData = np.concatenate(data_list, axis=2)
    mainData = binary_erosion(mainData, structure=structure).astype(mainData.dtype)

    labels_out = cc3d.connected_components(mainData, connectivity=26)
    valid_ids = np.unique(labels_out)
    masses = np.bincount(labels_out.ravel())[valid_ids]

    return np.mean(masses) if masses.size > 0 else 0

# Define the range for x, y, z dimensions
x_range = range(1, 5)  # Adjust the range as needed
y_range = range(1, 5)
z_range = range(1, 5)

best_structure = None
best_avg_mass_99 = -np.inf
best_avg_mass_50 = -np.inf

# Total combinations to process
total_combinations = len(list(product(x_range, y_range, z_range)))
current_combination = 0

# Iterate through all combinations of (x, y, z)
for x, y, z in product(x_range, y_range, z_range):
    current_combination += 1
    structure = np.ones((x, y, z))
    
    avg_mass_50 = calculate_average_mass(50, structure)
    avg_mass_99 = calculate_average_mass(99, structure)

    # Print progress
    print(f"Processing combination ({x}, {y}, {z}) - {current_combination}/{total_combinations}")

    # Check for valid average masses
    if avg_mass_50 is None or avg_mass_99 is None or np.isnan(avg_mass_50) or np.isnan(avg_mass_99):
        continue  # Skip this combination if mass calculation fails

    # Check if avg_mass_99 is greater than avg_mass_50
    if avg_mass_99 > avg_mass_50:
        print(f"Valid structure found: (x={x}, y={y}, z={z}) | Avg Mass 50: {avg_mass_50:.2f}, Avg Mass 99: {avg_mass_99:.2f}")
        
        # Update best structure if the current one is better
        if avg_mass_99 > best_avg_mass_99:
            best_structure = structure
            best_avg_mass_50 = avg_mass_50
            best_avg_mass_99 = avg_mass_99

# Output the best structure found
if best_structure is not None:
    print(f"Best structure sizes for binary erosion (x, y, z): ({best_structure.shape[0]}, {best_structure.shape[1]}, {best_structure.shape[2]})")
    print(f"Best average mass for frame 50: {best_avg_mass_50:.2f}, frame 99: {best_avg_mass_99:.2f}")
else:
    print("No valid structures found.")
