#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import logging
import csv
from PIL import Image
import numpy as np

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
        for row in reader:
            logging.info(f'train data : {row}')
            image_path = os.path.join(dataset_path, 'train_images', row['ImageId_ClassId'].split('_')[0])
            logging.info(f'image path : {image_path}')
            detect_image(image_path, row['EncodedPixels'])
            break

def detect_image(image_path, masks_pixels):
    if not os.path.isfile(image_path):
        raise f'{image_path} not found'
    logging.info(f'detect image: {image_path}')
    image = Image.open(image_path, 'r')
    w, h = image.size
    total = w * h
    logging.info(f'{image_path} width=${w}, height={h}, total={total}')
    masks = list(map(lambda x:int(x), masks_pixels.split(' ')))
    class Mask(object):
        def __init__(self, p, l):
            self.pos = p
            self.len = l
        def __repr__(self):
            return f'position: {self.pos}, length: {self.len}'
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
    pixels = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'L':
        channels = 1
    pixel_set = np.array(pixels).reshape((h, w, channels))
    logging.info(f'{pixel_set}')
    for m in mask_list:
        r = int(m.pos / w)
        c = int(m.pos % w)
        logging.info(f'{list(pixel_set)[r][c]}')

load_train_csv()

