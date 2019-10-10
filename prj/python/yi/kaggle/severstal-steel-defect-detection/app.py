#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import logging
import csv
from PIL import Image
import numpy as np
import cv2

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

dataset_path = '../../../../../dataset/kaggle/severstal-steel-defect-detection'

if not os.path.isdir(dataset_path):
    logger.error('dataset path not exist')
    sys.exit(1)
logger.info('dataset path found')

def load_train_csv():
    train_csv = os.path.join(dataset_path, 'train.csv')
    if not os.path.isfile(train_csv):
        raise 'train.csv not found'
    with open(train_csv, 'r') as train_fd:
        header = next(csv.reader(open(train_csv, 'r')))
        header_str = ', '.join(header)
        logging.info(f'train data header: {header_str}')
        reader = csv.DictReader(train_fd)
        n = 0
        for row in reader:
            logging.info(f'train data : {row}')
            image_name = row['ImageId_ClassId'].split('_')[0]
            image_path = os.path.join(dataset_path, 'train_images', image_name)
            logging.info(f'image path : {image_path}')
            if row['EncodedPixels'] != '':
                n += 1
            if n == 100:
                detect_image(image_name, image_path, row['EncodedPixels'])
                break


def detect_image(image_name, image_path, masks_pixels):
    if not os.path.isfile(image_path):
        raise f'{image_path} not found'
    logging.info(f'detect image: {image_path}')
    pixels = cv2.imread(image_path)
    h, w, c = pixels.shape
    total = w * h
    logging.info(f'{image_path} channle={c}, width={w}, height={h}, total={total}')
    masks = list(map(lambda x:int(x), masks_pixels.split(' ')))
    class Mask(object):
        def __init__(self, p, l):
            self.pos = p
            self.len = l
            self.end = p + l
        def __repr__(self):
            return f'position: {self.pos}, length: {self.len}'
        def start_and_end(self, rows):
            return ((int(self.pos / rows), int(self.pos % rows)),
                    (int(self.end / rows), int(self.end % rows)))
    mask_list = []
    i = 0
    while i < len(masks):
        p = masks[i]
        i += 1
        l = masks[i]
        i += 1
        m = Mask(p, l)
        mask_list.append(m)
        logging.info(f'{m}')
    for m in mask_list:
        start, end = m.start_and_end(h)
        logging.info(f'{start} - {end}')
        cv2.line(pixels, start, end, (202, 203, 151))
    cv2.imshow(image_name, pixels)
    cv2.waitKey(0)

load_train_csv()

