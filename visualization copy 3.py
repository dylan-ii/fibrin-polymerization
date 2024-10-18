import h5py
import numpy as np
import open3d as o3d
import cc3d
import math
import os

def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

directoryInString = "2023-07-27_120727\\thresholdedData\\"

directory = os.fsencode(directoryInString)

i=0

mainData = np.empty((20,2048,2048),dtype=bool)

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    print(filename)

    f = h5py.File(directoryInString+"\\"+filename)
    data = f['thresholdData'][145]

    mainData[i] = data

    i+=1

labels_out, N = cc3d.connected_components(mainData,connectivity=6,return_N=True,delta=0)
print(N)
print(labels_out.shape)
stats = cc3d.statistics(labels_out)

newCentroids = np.empty((stats['centroids'].shape))

for i,x in enumerate(stats['centroids']):
    newCentroids[i] = np.array([x[1],x[2],x[0]*15])

outputDirectoryInString = "2023-07-27_120727\\segmentedVoxels\\"
#os.mkdir(outputDirectoryInString)

for p,h in enumerate(labels_out):
    c = np.transpose(np.nonzero(h>=1))

    h5f = h5py.File(outputDirectoryInString+str(p), 'w')
    h5f.create_dataset('segmentedVoxels', data=c)
    h5f.close()

directoryInString = "2023-07-27_120727\\segmentedVoxels"
directory = os.fsencode(directoryInString)

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    #print(int(filename))

    f = h5py.File(directoryInString+"\\"+filename)
    data2 = f['segmentedVoxels']

    data2 = np.insert(data2, 2, int(filename), axis=1)

    i+=1

    try:
        mainData2
    except:
        mainData2 = data2
        continue
    else:  
        mainData2 = np.concatenate((mainData2, data2))

print("")

mainData3 = np.zeros((mainData2.shape[0],3))
mainData4 = np.zeros((N+1,3,3))

sets = 0

for dLayer, d in enumerate(labels_out):
    print(dLayer)
    #print("")
    #print(sets)
    #print("")
    for xLayer, x in enumerate(d):
        for yLayer, y in enumerate(x):
            if y == 0: continue
            dif = (np.array([dLayer,xLayer,yLayer])-mainData4[y][2])
            if dif[0] < 0 or dif[1] < 0 or dif[2] < 0:
                if magnitude(dif) > magnitude(mainData4[y][0]-mainData4[y][2]):
                    mainData4[y][0] = np.array([dLayer,xLayer,yLayer])
                    sets+=1
                    #print(mainData4[y][0]-stats['centroids'][y])
            else:
                #print(2)
                if magnitude(dif) > magnitude(mainData4[y][1]-mainData4[y][2]):
                    #print(mainData4[y][1]-stats['centroids'][y])
                    mainData4[y][1] = np.array([dLayer,xLayer,yLayer])
                    sets+=1

mainData5 = np.zeros((N+1,1))

for ind, x in enumerate(mainData4):
    mainData5[ind] = magnitude(x[0]-x[1])

print(np.average(mainData5))

outputDirectory2InString = "fiberLength.h5"

h5f = h5py.File(outputDirectory2InString, 'w')
h5f.create_dataset('fiberLength', data=mainData5)
h5f.close()