import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

#sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
#sam.to(device=DEVICE)

f = h5py.File("2023-07-27_152203\data\Cam_left_00000.lux.h5", 'r')

data = f['Data']
meta = f["metadata"]

for i, x in enumerate(data):
    if i > 25000:
        data3 = x

        data2 = data3

        plt.figure(figsize=(2048,1016))
        plt.imshow(data2)
        plt.axis('off')
        plt.show()

        for x in data2:
            #x[x < 50] = 0
            #x[x > 50] = 50
            x[x < 235] += 20
            x[x > 255] = 255

        data2 = data2.astype(np.uint8)
        data3 = data3.astype(np.uint8)

        data2 = cv.GaussianBlur(data2,(3,3),3)

        #cv.namedWindow('Raw',cv.WINDOW_NORMAL)
        #cv.imshow("Raw", data2)
        plt.figure(figsize=(2048,1016))
        plt.imshow(data2)
        plt.axis('off')
        print(i)
        #plt.show()
        #cv.resizeWindow('Raw', 2048,2048)

        #cv.imshow("Other Raw", data[110])

        #ret,th1 = cv.threshold(data2,155,255,cv.THRESH_BINARY)

        #cv.imshow("Binary Threshold", th1)

        #th2 = cv.adaptiveThreshold(data2,255,cv.ADAPTIVE_THRESH_MEAN_C,\
        #            cv.THRESH_BINARY,51,-20)

        #cv.imshow("Adaptive Thresh Mean", th2)

        th3 = cv.adaptiveThreshold(data2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY,51,-12)

        #T, th3 = cv.threshold(data2, 0, 255,
        #	cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        #th3 = np.invert(th3)
        #cv.namedWindow('Adaptive Guassian Threshold',cv.WINDOW_NORMAL)
        #cv.imshow("Adaptive Guassian Threshold", th3)

        plt.figure(figsize=(1016,2048))
        plt.imshow(th3)
        plt.axis('off')
        plt.show()

        #cv.resizeWindow('Adaptive Guassian Threshold', 2048,2048)

        #data2 = cv.cvtColor(th3, cv.COLOR_BGR2RGB)

        #mask_generator = SamAutomaticMaskGenerator(
        #    model=sam,
        #    points_per_side=64,
        #    pred_iou_thresh=0.3,
        #    stability_score_thresh=0.965,
        #    crop_n_layers=1,
        #    crop_n_points_downscale_factor=2,
        #    min_mask_region_area=0,  # Requires open-cv to run post-processing
        #)
        #result = mask_generator.generate(data2)

        #print(len(result))
        #
        #plt.figure(figsize=(1016,2048))
        #plt.imshow(data2)
        #show_anns(result)
        #plt.axis('off')
        #plt.show() 

        #blur = cv.GaussianBlur(data3,(5,5),0)
        #ret3,th4 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        #cv.imshow("OTSU Threshold", th4)
        #cv.waitKey(0)
