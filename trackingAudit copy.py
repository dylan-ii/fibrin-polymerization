import h5py
import numpy as np
import os
import cc3d
import open3d as o3d
import matplotlib.pyplot as plt

directoryInString = "2023-07-27_152203\\thresholdedData\\"
outputDirectoryInString = "2023-07-27_152203\\segmentedVoxels\\"
os.makedirs(outputDirectoryInString, exist_ok=True)

# Generate random distinct colors
def generate_distinct_colors(num_colors):
    return np.random.rand(num_colors, 3)  # Returns RGB colors

# Iterate over the desired time frames
for j in range(100):  # Normally 100
    print(f"Processing time frame: {j}")

    if j < 50: continue

    data_list = []  # List to hold data from each file

    # Load the thresholded data for the current time frame
    for file in os.listdir(directoryInString):
        filename = os.fsdecode(file)

        if (int(filename) - (int(filename) % 501)) / 501 == j:
            with h5py.File(os.path.join(directoryInString, filename), 'r') as f:
                data = f['thresholdData'][:]  # Load the dataset into a NumPy array
                if data.ndim == 2:  # Ensure data is 2D 
                    data = data[:, :, np.newaxis]  # Add a third dimension
                data_list.append(data)  # Append to the list

    # Concatenate the data into a 3D array
    if data_list:
        mainData = np.concatenate(data_list, axis=2)  # Stack along the third axis (depth)
    else:
        print("No data found for this time frame.")
        continue  # Skip to the next frame if no data

    from scipy.ndimage import binary_erosion

    # Apply binary erosion
    mainData = binary_erosion(mainData, structure=np.ones((4, 4, 4))).astype(mainData.dtype)

    # Perform connected components analysis
    labels_out, N = cc3d.connected_components(mainData, connectivity=6, return_N=True, delta=0)
    labels_out = cc3d.dust(labels_out, connectivity=6, threshold=375)

    print("Segments found.")

    # Extract non-zero voxel coordinates and their segment IDs
    non_zero_voxels = np.argwhere(labels_out > 0)
    segment_ids = labels_out[non_zero_voxels[:, 0], non_zero_voxels[:, 1], non_zero_voxels[:, 2]]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(non_zero_voxels)

    # Generate distinct colors
    distinct_colors = generate_distinct_colors(N)

    # Map segment IDs to distinct colors
    colors = distinct_colors[segment_ids]  # Map segment IDs to colors
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(distinct_colors.shape)

    # Visualize all segments for the current frame
    o3d.visualization.draw_geometries([pcd], window_name=f'Segmented Fibers - Frame {j}')

print("Processing complete.")
