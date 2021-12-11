#!/usr/bin/env python
# -*- codiing:utf-8 -*-
import os
import math
import numpy as np
import cv2

dataset_dir = os.path.join('../..', 'dataset')
test_image = os.path.join(dataset_dir, 'test.jpg')
print("==== Start ====")
# test_image = os.path.join('/Users/yixiaobin/Downloads', 'test.jpg')
assert(os.path.isfile(test_image))

img = cv2.imread(test_image, 0)
cv2.imshow('origin image', img)
cv2.waitKey(0)
print(img)
print(type(img))

# convoluation with 3 x 3 core matrix
print(img.shape)
# (cols, rows) = image.shpae
# for n in rows:
#     for m in cols:
#         print(image[n][m])
#         break
# core = np.array(list(1.0/9 for i in range(0, 9))).reshape(3, 3)
core_num = 5
core = np.ones([core_num, core_num])
print(core)

for e in img.flat:
    print(e)
    break


def get_data_from_image(img, core_num, row, col, rows, cols):
    data = []
    for n in range(math.ceil(-core_num/2.0), math.ceil(core_num/2)):
        for m in range(math.ceil(-core_num/2.0), math.ceil(core_num/2)):
            r = row + n
            c = col + m
            if r < 0 or r >= rows or c < 0 or c >= cols:
                data.append(0)
            else:
                data.append(img[r * cols + c])
    return np.array(data).reshape(core_num, core_num)


print('==== get_data_from_image ====')
for i in range(0, 20):
    data = get_data_from_image(img.flat, core_num, i, i, img.shape[0], img.shape[1])
    print(data)

pass


def point_mul(img_data, core_num, core):
    d = 0
    for n in range(math.ceil(-core_num/2.0), math.ceil(core_num/2)):
        for m in range(math.ceil(-core_num/2.0), math.ceil(core_num/2)):
            d += img_data[n][m] * core[n][m]
    r = int(d / (core_num * core_num))
    if r < 0:
        print(f'==== Error: ${r} ====')
    return r


# point mul from 0 to 20
print('==== point mul ====')
for i in range(0, 20):
    data = get_data_from_image(img.flat, core_num, i, i, img.shape[0], img.shape[1])
    mul = point_mul(data, core_num, core)
    print(mul)

new_img = []
for n in range(0, img.shape[0]):
    for m in range(0, img.shape[1]):
        data = get_data_from_image(img.flat, core_num, n, m, img.shape[0], img.shape[1])
        mul = point_mul(data, core_num, core)
        new_img.append(mul)

new_img_mat = np.array(new_img, dtype=np.uint8).reshape(img.shape[0], img.shape[1])
print(new_img_mat)
print('==== show low-pass filter image, bulr origin image ====')
cv2.imshow('low-pass filter image shower', new_img_mat)
cv2.waitKey(0)
