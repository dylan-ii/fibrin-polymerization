import h5py
import numpy as np
import cv2 as cv
import os

# this program is very similar to that of threshold generator for real volumes, with a few changes optimized to test volumes
# the two major differences are a decrease instead of increase in global luminosity (yields more accurate thresholds for test volumes)
# and changed iteration over main data as the organization of raw test volume data is different. 
# instead of a single h5 file with 50100 depth layers to account for all time and depth, test volumes are ordered into folders by depth with h5 files to account for all times at that depth

directoryInString = "2023-07-27_120727\data\\"
outputDirectoryInString = "2023-07-27_120727\\thresholdedData\\"
directory = os.fsencode(directoryInString)

def getArrayThresh(inputFile, d):
    f = h5py.File(inputFile)

    data = f['Data'][d]

    for x in data:
        x[x < 50] = 0
        x[x > 50] -= 50
        x[x > 255] = 255
    
    data = data.astype(np.uint8)
    data = cv.medianBlur(data,7)

    th = cv.adaptiveThreshold(data,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,51,-20)

    return th

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    print(filename)

    f = h5py.File(directoryInString+filename+"\Cam_left_00000.lux.h5")
    data = f['Data']
    fileData = np.zeros((291,2048,2048),dtype=bool)

    for i in range(0, data.shape[0]):
        currentArray = getArrayThresh(directoryInString+filename+"\Cam_left_00000.lux.h5", i)
        fileData[i] = currentArray
        if i%100 == 0:
            print(np.round(100*(i/data.shape[0]),0))

    h5f = h5py.File(outputDirectoryInString+filename, 'w')
    h5f.create_dataset('thresholdData', data=fileData.astype(bool),compression='gzip')
    h5f.close()