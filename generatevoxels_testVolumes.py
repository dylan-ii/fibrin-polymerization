import h5py
import numpy as np
import os

# test volumes have different numbers of depth layers compared to real volumes. kept these files separate so that I could always use whichever program on both

directoryInString = "2023-07-27_132735\\thresholdedData\\"
outputDirectoryInString = "2023-07-27_132735\\thresholdedVoxels\\"

directory = os.fsencode(directoryInString)

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    f = h5py.File(directoryInString+filename)
    data = f['thresholdData']

    os.mkdir(outputDirectoryInString)
    outputDirectoryInString = "2023-07-27_132735\\thresholdedVoxels\\"+filename+"\\"


    for p,h in enumerate(data):
        c = np.transpose(np.nonzero(h==True))

        h5f = h5py.File(outputDirectoryInString+str(p), 'w')
        h5f.create_dataset('thresholdedVoxels', data=c)
        h5f.close()