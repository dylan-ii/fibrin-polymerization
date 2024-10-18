import h5py
import numpy as np
import open3d as o3d
import cc3d
import matplotlib.pyplot as plt
import pandas as pd
import os
import trackpy as tp
from skimage.measure import label, regionprops
import time

directoryInString = "2023-07-27_152203\\thresholdedData\\"
directory = os.fsencode(directoryInString)

outputDirectoryInString = "2023-07-27_152203\\segmentedVoxels\\"
#os.mkdir(outputDirectoryInString)

z=0

outputFinal = None

coverage = np.zeros((100))
labels_out = None

for j in range(100): #normally 100

    #if j < 53: continue

    mainData = None

    for file in os.listdir(directory):

        filename = os.fsdecode(file)

        if (int(filename)-(int(filename)%501))/501 == j:

            f = h5py.File(directoryInString+"\\"+filename)
            data = f['thresholdData']

            if mainData is None:
                mainData = data
            else:  
                mainData = np.dstack((mainData, data))

    print(mainData.shape)
    print(j)

    #features = tp.locate(mainData, diameter=(9,21,21), separation=(3, 11, 11))
    #print(features.head())

    #tp.subpx_bias(features)
    #tp.annotate3d(features, mainData)
    #print('Features found: {0}'.format(len(features)))

    #plt.show()

    labels_out1, N = cc3d.connected_components(mainData,connectivity=18,return_N=True, delta=0)#, connectivity=6,return_N=True,delta=0)
    labels_out = cc3d.dust(labels_out1,connectivity=6,threshold=100)

    # Define minimum size threshold
    #min_size = 100  # Adjust based on your needs

    #labels_out = filtered_labels

    #print(labels_out.shape)

    stats = cc3d.statistics(labels_out)

    #print(type(stats['centroids']))
    print(stats['centroids'].shape)

    stats['centroids'] = stats['centroids'][~np.isnan(stats['centroids']).any(axis=1)]

    print(stats['centroids'].shape)

    #eatures = tp.locate(mainData, diameter=(43))

    #print(features.head())
    #centroids = pd.DataFrame({"x": stats['centroids'][:, 0], "y": stats['centroids'][:, 1], "z": stats['centroids'][:, 2], "frame": 0})

    #print(centroids.head())

    #print(type(features))
    #print(type(centroids))

    #print(features.shape)
    #print(centroids.shape)

    #tp.annotate3d(centroids, mainData)
    #print('Features found: {0}'.format(len(centroids)))

    #plt.show()

    #os.mkdir(outputDirectoryInString+str(j))

    for p,h in enumerate(labels_out):
        c = np.transpose(np.nonzero(h>=1))

        h5f = h5py.File(outputDirectoryInString+str(j)+"\\"+str(p), 'w')
        h5f.create_dataset('segmentedVoxels', data=c)
        h5f.close()

    print(labels_out.shape)
#    stats = cc3d.statistics(labels_out)
#    print(np.transpose(np.nonzero(labels_out!=0)).shape[1])
#    print(np.transpose(np.nonzero(labels_out!=0)).shape[0])
#    print((np.transpose(np.nonzero(labels_out!=0)).shape[0])/(2048*1016*501))
#    coverage[j] = (np.transpose(np.nonzero(labels_out!=0)).shape[0])/(2048*1016*501)
#    print(coverage[j])

#    print(coverage)

#    print(j)

#    if j == 5:
#        plt.style.use("seaborn-v0_8-darkgrid")
#        plt.plot(coverage)
#        plt.title('Title', fontsize=40)
#        plt.xlabel('Time', fontsize=20)
#        plt.ylabel('Intensity', fontsize=20)
#        plt.show()

#plt.style.use("seaborn-v0_8-darkgrid")
#plt.plot(coverage)
#plt.title('Title', fontsize=40)
#plt.xlabel('Time', fontsize=20)
#plt.ylabel('Intensity', fontsize=20)
#plt.show()

#np.save("coverage.npy",coverage)

#newCentroids = np.empty((stats['centroids'].shape))

#for i,x in enumerate(stats['centroids']):
#    newCentroids[i] = np.array([x[1],x[2],x[0]])

#print(newCentroids.shape)

directoryInString = "2023-07-27_152203\\segmentedVoxels"
directory = os.fsencode(directoryInString)

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    #print(int(filename))

    f = h5py.File(directoryInString+"\\"+filename)
    data2 = f['segmentedVoxels']

    #print(data2.shape)

    data2 = np.insert(data2, 2, int(filename), axis=1)

    #print(data2[int(filename)][100][100])

    i+=1

    try:
        mainData2
    except:
        mainData2 = data2
        continue
    else:  
        mainData2 = np.concatenate((mainData2, data2))

print("")
print(mainData2.shape)

print(labels_out.shape)

mainData3 = np.zeros((mainData2.shape[0],3))
rgb_t = np.zeros((mainData2.shape[0],3))

z = 0

weights = np.empty((N+1,4))
distribution = np.zeros((N+1), dtype=np.int32)
rng = np.random.default_rng(58688)

def randomWeight(index2):
    if weights[index2][0]: return weights[index2]
    else: weights[index2] = [.5+(rng.random()/2), .5+(rng.random()/2), .5+(rng.random()/2), rng.random()+.35]; return weights[index2]

weightCount = 0

for index,h in enumerate(mainData2):
    if index%100000 == 0: print(index)#print(i); print(mainData2.shape)
    #print(labels_out[h[0-1]][h[1]][h[2]])
    #print(labels_out[h[0]][h[1]][h[2]])
    #print(mainData2[i])
    #print(mainData2)
    if type(h) != np.int64:
        #print((labels_out[h[0-1]][h[1]][h[2]])/N)
        #print(h)
            #print(randomWeight(labels_out[h[2]][h[0]][h[1]]-1))
        #print(labels_out[h[2]][h[0]][h[1]])
        distribution[labels_out[h[2]][h[0]][h[1]]] += 1
        #print(distribution[labels_out[h[2]][h[0]][h[1]]])
        a = randomWeight(labels_out[h[2]][h[0]][h[1]])
        b = a[3]
        rgb_t[index] = [a[0],a[1],a[2]]
        weightCount+=a[0]

for ind, x in enumerate(mainData2):
    mainData3[ind] = np.array([x[2],x[0],x[1]])

print(distribution)

plt.style.use("seaborn-v0_8-darkgrid")
plt.hist(distribution, bins=20,label=["40"])
plt.legend(loc="upper right")
plt.show()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(mainData3)
pcd.colors = o3d.utility.Vector3dVector(rgb_t)
o3d.visualization.draw_geometries([pcd])

#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(newCentroids)
#o3d.visualization.draw_geometries([pcd])
