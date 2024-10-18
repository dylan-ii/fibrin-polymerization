import h5py
import numpy as np
import os
import cc3d
import open3d as o3d
from scipy.ndimage import binary_erosion

directoryInString = "2023-07-27_152203\\thresholdedData\\"
outputDirectoryInString = "2023-07-27_152203\\segmentedVoxels\\"
imagesDirectory = os.path.join(outputDirectoryInString, "images")
os.makedirs(imagesDirectory, exist_ok=True)

# Generate random distinct colors
def generate_distinct_colors(num_colors):
    return np.random.rand(num_colors, 3)  # Returns RGB colors

# Set fixed bounds for the visualization
def set_view_bounds(vis):
    vis.get_view_control().set_lookat([1016 / 2, 2048 / 2, 501 / 2])
    vis.get_view_control().set_front([0, 0, -1])  # Looking from the front
    vis.get_view_control().set_up([0, 1, 0])      # Up direction
    vis.get_view_control().set_zoom(0.5)           # Adjust zoom to fit the volume

# Iterate over the desired time frames
for j in range(100):  # Normally 100
    print(f"Processing time frame: {j}")

    #if j < 98: continue

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

    # Apply binary erosion
    mainData = binary_erosion(mainData, structure=np.ones((3, 6, 3))).astype(mainData.dtype)

    # Perform connected components analysis
    labels_out, N = cc3d.connected_components(mainData, connectivity=26, return_N=True, delta=1)
    labels_out = cc3d.dust(labels_out, connectivity=26, threshold=375)

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

    # Set up the visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Create window but keep it hidden

    vis.add_geometry(pcd)
    vis.update_geometry(pcd)

    # Set the view bounds to cover the entire volume
    set_view_bounds(vis)

    # Capture the current frame and save as an image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(imagesDirectory, f'segmented_fibers_frame_{j}.png'))

    # Close the visualizer
    vis.destroy_window()

print("Processing complete.")
