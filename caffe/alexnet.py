#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from header import *
import cv2
import lmdb

ROOT_DIR = os.path.dirname(__file__)
PYPB2_FILE = os.path.join(ROOT_DIR, 'alexnet_pb2.py')
if os.path.exists(PYPB2_FILE):
  os.remove(PYPB2_FILE)

shell(f'protoc -I={ROOT_DIR} --python_out={ROOT_DIR} {ROOT_DIR}/alexnet.proto', waiting=True, cwd=ROOT_DIR, valued=True)
time.sleep(1)
from alexnet_pb2 import *

class ImageUtils:
  @staticmethod
  def load(path, flags=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(path, flags)
    if img is None:
        logger.error(f'load image not found {path}')
    return img

  @staticmethod
  def show(img, title='img'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
  
  @staticmethod 
  def load_from_datum(datum):
    if len(datum.data):
      return np.fromstring(datum.data, dtype=np.uint8).reshape(datum.height, datum.width, datum.channels)
    return np.array(datum.float_data).astype(float).reshape(datum.height, datum.width, datum.channels)

class AlexNet:
  class DB:
    def __init__(self, **kwargs):
      self.filepath = os.path.join(ROOT_DIR, kwargs['filepath']) if 'filepath' in kwargs else os.path.join(ROOT_DIR, 'mnist', 'mnist_train_lmdb')
      self.input = Input()
    
    def setup(self):
      self.db = lmdb.open(self.filepath)
      txn = self.db.begin()
      self.cursor = txn.cursor()
    
    def load(self):
      i = 0
      for key, value in self.cursor:
        if i == self.input.branch_size * 96:
          break
        if i % self.input.branch_size == 0:
          if i != 0:
            self.input.branches.append(branch)
            self.input.size += 1
          branch = Branch()
        data = Datum()
        data.ParseFromString(value)
        branch.datas.append(data)
        i += 1
       
  class Layer:
    def __init__(self, **kwargs):
      pass
    
    def setup():
      pass

  def __init__(self, **kwargs):
    self.db = AlexNet.DB(**kwargs)
    self.db.setup()  
  
        
      
class AlexNetUnitTests(UnitTests):
  def __init__(self):
    super().__init__(__file__)

  # @UnitTests.skip
  def init_test(self):
    alexnet = AlexNet()
    assert alexnet.db is not None
    alexnet.db.load()
    logger.debug(alexnet.db.input.branch_size)
    logger.debug(alexnet.db.input.size)
    assert alexnet.db.input.size == len(alexnet.db.input.branches)
    branch_images = []
    for i in range(alexnet.db.input.size):
      branch = alexnet.db.input.branches[i]
      images = []
      for j in range(branch.size):
        weight_data = branch.datas[j]
        images.append(ImageUtils.load_from_datum(weight_data))
      branch_images.append(np.hstack(images))
    ImageUtils.show(np.vstack(branch_images))
      

AlexNetUnitTests().run()
