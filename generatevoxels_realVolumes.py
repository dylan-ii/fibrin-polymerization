import h5py
import numpy as np
import os

# I've tried to make this as compact and quick as possible but there are probably ways of speeding it up

directoryInString = "2023-07-27_152203\\thresholdedData\\"
outputDirectoryInString = "2023-07-27_152203\\thresholdedVoxels\\"

directory = os.fsencode(directoryInString)

# make output folders for every depth layer, data organized by h5 files under a specific depth layer for each time instance
# may refactor this into folders for every time instance, with h5 files for each depth layer or a single h5 file for every time instance.
# that refactoring would assist in easing the loading process which always requires weird transposing under current methods

for i in range(501):
    os.mkdir(outputDirectoryInString+str(i))

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    print(int(filename))
    voxelLayer = int(filename)%501

    f = h5py.File(directoryInString+filename)
    data = f['thresholdData']

    outputDirectoryInString = "2023-07-27_152203\\thresholdedVoxels\\"+str(voxelLayer)+"\\"

    c = np.transpose(np.nonzero(data[:]==True))

    h5f = h5py.File(outputDirectoryInString+str(int((int(filename)-1)/501)), 'w')
    h5f.create_dataset('thresholdedVoxels', data=c)
    h5f.close()