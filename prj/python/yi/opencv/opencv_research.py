#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

dataset_path = '../../../../dataset/opencv'


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

def lookup_opencv_flags():
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    print('\n'.join(flags))

def filter_with_bitwise(img1_file, img2_file):
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)  # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# show_image(os.path.join(dataset_path, 'mesi.png'))
# draw()
# lookup_opencv_flags()
filter_with_bitwise(os.path.join(dataset_path, 'mesi.png'), os.path.join(dataset_path, 'logo.png'))