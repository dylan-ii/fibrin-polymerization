import h5py
import numpy as np
import open3d as o3d
import cc3d
import math
import os

def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))
j=0
smallerDirectory = "2023-07-27_120727\\thresholdedVoxels\\stack_1-x00-y00-Tile_channel_3-642_obj_left\\"
dir = os.fsencode(smallerDirectory)

a = 0
b = 0

for x in os.listdir(dir):
    directoryInString = "2023-07-27_120727\\thresholdedData\\"
    directory = os.fsencode(directoryInString)

    i=0

    mainData = np.empty((20,2048,2048),dtype=bool)

    for file in os.listdir(directory):

        filename = os.fsdecode(file)

        f = h5py.File(directoryInString+"\\"+filename)
        data = f['thresholdData'][j]

        mainData[i] = data

        i+=1

    labels_out, N = cc3d.connected_components(mainData,connectivity=6,return_N=True)
    stats = cc3d.statistics(labels_out)

    #print((np.sum(stats["voxel_counts"])-np.max(stats["voxel_counts"]))*(.195**2))

    #if j == 0:
    #    a = np.sum(stats["voxel_counts"])-np.max(stats["voxel_counts"])
    #elif j == 50:
    #    b = np.sum(stats["voxel_counts"])-np.max(stats["voxel_counts"])
    #    print(b-a)

    outputDirectoryInString = "2023-07-27_120727\\segmentedVoxels\\"

    for p,h in enumerate(labels_out):
        c = np.transpose(np.nonzero(h>=1))

        h5f = h5py.File(outputDirectoryInString+str(p), 'w')
        h5f.create_dataset('segmentedVoxels', data=c)
        h5f.close()

    directoryInString = "2023-07-27_120727\\segmentedVoxels"
    directory = os.fsencode(directoryInString)

    outputDirectory2InString = "fiberVoxels\\"+str(j)+".h5"

    h5f = h5py.File(outputDirectory2InString, 'w')
    h5f.create_dataset('fiberVoxels', data=stats["voxel_counts"])
    h5f.close()
    print(j)
    j+=1