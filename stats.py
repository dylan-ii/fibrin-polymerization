import h5py
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

smallerDirectory = "fiberVoxels"
dir = os.fsencode(smallerDirectory)

mainArray = np.zeros((291,1))

for file in os.listdir(dir):

    filename = os.fsdecode(file)

    #print(filename)

    f = h5py.File(smallerDirectory+"\\"+filename)
    data = f['fiberVoxels']

    if filename == "0.h5": print(np.sum(data)-np.max(data))

    mainArray[int(filename.replace(".h5",""))] = (np.sum(data)-np.max(data))
    if filename == "0.h5": print(mainArray[0])

print(np.sum(mainArray[0]))

plt.plot(mainArray)
plt.show()