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

    print(stats["voxel_counts"])

    outputDirectoryInString = "2023-07-27_120727\\segmentedVoxels\\"

    for p,h in enumerate(labels_out):
        c = np.transpose(np.nonzero(h>=1))

        h5f = h5py.File(outputDirectoryInString+str(p), 'w')
        h5f.create_dataset('segmentedVoxels', data=c)
        h5f.close()

    directoryInString = "2023-07-27_120727\\segmentedVoxels"
    directory = os.fsencode(directoryInString)

    hasBeenSet = False

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        f = h5py.File(directoryInString+"\\"+filename)
        data2 = f['segmentedVoxels']

        data2 = np.insert(data2, 2, int(filename), axis=1)

        i+=1

        if hasBeenSet == False:
            mainData2 = data2
            hasBeenSet = True
            continue
        else:  
            mainData2 = np.concatenate((mainData2, data2))

    mainData3 = np.zeros((mainData2.shape[0],3))
    mainData4 = np.zeros((N+1,3,3))

    sets = 0

    for dLayer, d in enumerate(labels_out):
        #print(dLayer)
        for xLayer, x in enumerate(d):
            for yLayer, y in enumerate(x):
                if y == 0: continue
                dif = (np.array([dLayer,xLayer,yLayer])-mainData4[y][2])
                if dif[0] < 0 or dif[1] < 0 or dif[2] < 0:
                    if magnitude(dif) > magnitude(mainData4[y][0]-mainData4[y][2]):
                        mainData4[y][0] = np.array([dLayer,xLayer,yLayer])
                        sets+=1
                else:
                    if magnitude(dif) > magnitude(mainData4[y][1]-mainData4[y][2]):
                        mainData4[y][1] = np.array([dLayer,xLayer,yLayer])
                        sets+=1

    mainData5 = np.zeros((N+1,1))

    for ind, x in enumerate(mainData4):
        mainData5[ind] = magnitude(x[0]-x[1])

    outputDirectory2InString = "fiberLength\\"+str(j)+".h5"

    h5f = h5py.File(outputDirectory2InString, 'w')
    h5f.create_dataset('fiberLength', data=mainData5)
    h5f.close()
    print(j)
    j+=1