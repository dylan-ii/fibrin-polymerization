import h5py
import numpy as np
import open3d as o3d

import os

directoryInString = "2023-07-27_132735\\thresholdedVoxels\\"
directory = os.fsencode(directoryInString)

i=0
for file in os.listdir(directory):

    filename = os.fsdecode(file)

    print(filename)

    f = h5py.File(directoryInString+"\\"+filename+"\\"+"101")#+"\\"+"290")
    data = f['thresholdedVoxels']

    data = np.insert(data, 2, i, axis=1)

    i+=1

    try:
        mainData
    except:
        mainData = data
        continue
    else:  
        mainData = np.concatenate((mainData, data))

print(mainData.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(mainData)
o3d.visualization.draw_geometries([pcd])