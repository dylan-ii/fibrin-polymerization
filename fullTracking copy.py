import h5py
import numpy as np
import pandas as pd
import os
import trackpy as tp
import cc3d
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import binary_erosion

directoryInString = "2023-07-27_152203\\thresholdedData\\"
outputDirectoryInString = "2023-07-27_152203\\segmentedVoxels\\"
centroidsOutputDirectory = os.path.join(outputDirectoryInString, "centroids\\")
os.makedirs(centroidsOutputDirectory, exist_ok=True)

# Iterate over the desired time frames
for j in range(100):  # Normally 100

    if j != 50 and j != 99: continue
 
    print(f"Processing time frame: {j}")

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
        print(mainData.shape)
    else:
        print("No data found for this time frame.")
        continue  # Skip to the next frame if no data

    #if j >= 15:
    # Apply binary erosion
    mainData = binary_erosion(mainData, structure=np.ones((4, 4, 1))).astype(mainData.dtype)

    # Perform connected components analysis
    labels_out1, N = cc3d.connected_components(mainData, connectivity=6, return_N=True, delta=0)
    labels_out = cc3d.dust(labels_out1, connectivity=6, threshold=375)

    # Calculate statistics including centroids
    stats = cc3d.statistics(labels_out)
    centroids = stats['centroids'][~np.isnan(stats['centroids']).any(axis=1)]

    # Calculate mass based on voxel counts using np.bincount
    masses = np.bincount(labels_out.ravel())  # Count the occurrences of each label

    # Prepare mass for only the existing centroids
    valid_ids = np.unique(labels_out)
    valid_masses = masses[valid_ids]  # Get masses for valid IDs

    # Match masses to centroids
    mass_array = np.zeros(len(centroids))
    for i, centroid in enumerate(centroids):
        centroid_id = i + 1  # Assuming IDs start from 1
        mass_array[i] = valid_masses[centroid_id] if centroid_id < len(valid_masses) else 0

    print(mass_array[0])
    print(np.mean(mass_array))

    print(f"Centroids shape: {centroids.shape}, Mass shape: {mass_array.shape}")

    # Store centroids to a CSV file including sizes
    if centroids.size > 0:
        centroids_df = pd.DataFrame({
            "x": centroids[:, 0],
            "y": centroids[:, 1],
            "z": centroids[:, 2],
            "mass": mass_array,  # Use mass calculated from voxel counts
            "frame": j
        })
        centroids_df.to_csv(os.path.join(centroidsOutputDirectory, f"centroids_frame_{j}.csv"), index=False)
        print(f"Saved {len(centroids)} centroids to {centroidsOutputDirectory}centroids_frame_{j}.csv")
    else:
        print(f"No centroids found for frame {j}.")

# Load all centroid files for tracking
print("Processing complete. Now running TrackPy for tracking.")
all_centroids_df = pd.concat(
    (pd.read_csv(os.path.join(centroidsOutputDirectory, f)) for f in os.listdir(centroidsOutputDirectory)),
    ignore_index=True
)

# Use TrackPy to link centroids with mass input
search_range = 15
features = tp.link_df(all_centroids_df, search_range=search_range, memory=2)
print(f'Features tracked: {len(features)}')

# Initialize a list to store mass distributions for each frame
mass_distributions = []

# Calculate mass distribution for each frame
for j in range(100):  # Change this range as needed based on your frames
    # Load the centroids for the current frame
    centroids_frame_df = pd.read_csv(os.path.join(centroidsOutputDirectory, f"centroids_frame_{j}.csv"))
    
    # Store the mass distribution
    mass_distributions.append(centroids_frame_df['mass'].values)

# Define the common x-axis limits based on observed data
# Using 99th percentile to limit the x-axis for better visibility
max_mass = np.percentile(np.concatenate(mass_distributions), 99)
common_x_limits = (0, max_mass + 1)  # +1 for some breathing room

# Find the maximum frequency across all frames for consistent y-axis scaling
max_frequency = 0
for masses in mass_distributions:
    if len(masses) > 0:
        counts, _ = np.histogram(masses, bins=30, range=common_x_limits)
        max_frequency = max(max_frequency, counts.max())

# Loop to create and save histograms for each frame
for frame_index, masses in enumerate(mass_distributions):
    plt.figure(figsize=(10, 6))
    
    # Create the histogram
    plt.hist(masses, bins=30, range=common_x_limits, alpha=0.7, color='blue', edgecolor='black')
    
    # Customize the plot
    plt.xlim(common_x_limits)
    plt.ylim(0, max_frequency)  # Set y-axis limits to maximum frequency
    plt.xlabel('Mass of Tracked Fibers')
    plt.ylabel('Frequency')
    plt.title(f'Mass Distribution of Tracked Fibers - Frame {frame_index}')
    plt.grid(True)
    
    # Save the histogram as a PNG file
    plt.savefig(os.path.join(outputDirectoryInString, f'histogram_mass_distribution_frame_{frame_index}.png'))
    plt.close()  # Close the figure to free memory

# Count frames for each tracked object
frame_counts = features['particle'].value_counts()
filtered_ids = frame_counts[frame_counts > 5].index  # Keep only those that exist in multiple frames

print(len(filtered_ids))

# Gather Last Frames and Track Durations
last_frames = features.groupby('particle')['frame'].max().values
first_frames = features.groupby('particle')['frame'].min().values
num_frames_tracked = last_frames - first_frames

# Create a DataFrame for easy plotting
tracking_data = pd.DataFrame({
    'last_frame': last_frames,
    'num_frames_tracked': num_frames_tracked
})

# Plotting in 2D
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    tracking_data['last_frame'],
    tracking_data['num_frames_tracked'],
    s=50,  # Scale point size by tracking duration
    c=tracking_data['num_frames_tracked'],  # Color by tracking duration
    cmap='viridis',  # Color map
    alpha=0.6
)

# Add color bar to indicate tracking duration
cbar = plt.colorbar(scatter)
cbar.set_label('Number of Frames Tracked')

plt.xlabel('Last Frame')
plt.ylabel('Number of Frames Tracked')
plt.title('2D Visualization of Tracked Particles')
plt.grid(True)
plt.show()

# Filter for objects tracked longer than 10 frames
filtered_tracking_data = tracking_data[tracking_data['num_frames_tracked'] > 10]

# Count the frequency of drops at each frame
drop_counts = filtered_tracking_data['last_frame'].value_counts().sort_index()

# Plotting the drop counts
plt.figure(figsize=(10, 6))
plt.bar(drop_counts.index, drop_counts.values, color='orange', alpha=0.7)
plt.xlabel('Frame at Which Object was Dropped')
plt.ylabel('Count of Objects Dropped')
plt.title('Count of Tracked Objects Dropped by Frame (Tracked > 10 Frames)')
plt.xticks(np.arange(0, 100, 5))  # Optional: Set x-ticks for better readability
plt.grid(axis='y')
plt.show()

# Calculate average mass for each frame
# Assuming mass is part of features DataFrame (you might need to adjust based on your structure)
# First, ensure you have mass included in the features DataFrame
features['mass'] = features['mass'].fillna(0)  # Replace NaNs if necessary

average_mass_per_frame = features.groupby('frame')['mass'].mean().reset_index()

# Step 7: Plotting the average mass per frame
plt.figure(figsize=(10, 6))
plt.plot(average_mass_per_frame['frame'], average_mass_per_frame['mass'], marker='o', color='blue', label='Average Mass')
plt.xlabel('Frame')
plt.ylabel('Average Mass of Tracked Fibers')
plt.title('Average Mass of Tracked Fibers Over Each Frame')
plt.grid(True)
plt.legend()
plt.show()

# Visualization of positions for a random set of tracked objects over the time steps
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Select a random sample of tracked objects that exist in multiple frames
sample_ids = np.random.choice(filtered_ids, size=min(1, len(filtered_ids)), replace=False)  # Adjust sample size as needed

# Plot the positions of the selected objects over time
for particle_id in sample_ids:
    particle_features = features[features['particle'] == particle_id]
    print(particle_features)
    # Plot each position as a point
    ax.scatter(particle_features['x'], particle_features['y'], particle_features['z'], label=f'Particle {particle_id}', s=50)

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('Positions of Randomly Selected Tracked Objects Over Time')
ax.legend()
plt.show()
