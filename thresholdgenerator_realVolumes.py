import h5py
import numpy as np
import cv2 as cv
import os

# had the goal of making this as compact and efficient as I could
# inputs raw h5 data from real volumes (which are single h5 files), and ouputs h5 files of thresholded data

directoryInString = "2023-07-27_132735\data\stack_0_channel_0-Line 0.5 fgn 0.1_obj_left\Cam_left_00000.lux.h5"
outputDirectoryInString = "2023-07-27_132735\\thresholdedData\\"
directory = os.fsencode(directoryInString)

# function for getting exact thresholds for specific inputfile at depth layer d

def getArrayThresh(inputFile, d):
    f = h5py.File(inputFile)

    data = f['Data'][d] 

    # process data into 8 bit integers by capping at 255, bring up global luminosity by 25
    for x in data:
        x[x < 25] = 0
        x[x > 25] += 25
        x[x > 255] = 255
    
    # formally declare as 8 bit int, gaus blur
    data = data.astype(np.uint8)
    data = cv.GaussianBlur(data,(3,3),3)

    # use open cv adaptive gaus for final thresh and return
    th = cv.adaptiveThreshold(data,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY,51,-15)

    return th

filename = os.fsdecode(directoryInString)

f = h5py.File(filename)
data = f['Data']

dataForLayer = np.zeros((1016,2048), dtype=bool)

# iterate over depth/time layers in h5, export all thresholded data
# *relevant note*: for real volumes, h5 is 3d with 50100 z layers
# all are depth, ordered by time, with 501 depth layers per time step (ex: layer 572 in h5 is depth layer 70 of time 2)
for j,x in enumerate(data):
    if j%100 == 0:
        print(j)

    fileData = np.zeros((1016,2048),dtype=bool)
    currentArray = getArrayThresh(directoryInString, j)
    fileData = currentArray
    
    h5f = h5py.File(outputDirectoryInString+str(j), 'w')
    h5f.create_dataset('thresholdData', data=fileData.astype(bool),compression='gzip')
    h5f.close()