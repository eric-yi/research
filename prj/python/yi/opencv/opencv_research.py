#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

dataset_path = '../../../../dataset'


def show_image(img_file):
    img = cv2.imread(img_file, 0)
    # plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.imshow(img)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


img = np.zeros((512, 512, 3), np.uint8)
drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

def draw():
    global ix, iy, drawing, mode, img
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while True:
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    cv2.destroyAllWindows()

# show_image(os.path.join(dataset_path, 'kaggle/severstal-steel-defect-detection/test_images', 'f04940e2b.jpg'))
draw()