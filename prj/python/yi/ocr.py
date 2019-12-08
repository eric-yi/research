#!/usr/bin/env python

# -*- coding:utf-8 -*-

import pytesseract
from PIL import Image

image_path = '../../../dataset/image/test.png'
image = Image.open(image_path);
text = pytesseract.image_to_string(image, lang='eng')
print(text)