#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    import caffe
except Exception as e:
    print(e)
    

plt.rcParams['figure.figsize'] = (10, 10)    
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

root_dir = os.path.dirname(__file__)

model = os.path.join(root_dir, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
model_def = os.path.join(root_dir, 'models/bvlc_reference_caffenet/deploy.prototxt')
model_weights = os.path.join(root_dir, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

if os.path.isfile(model):
    print('CaffeNet Model Found.')
else:
    print('CaffeNet Model Not Found.')
    
caffe_root = '/opt/caffe'       # in docker container

caffe.set_mode_cpu()    
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

mu = np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

# image = caffe.io.load_image(os.path.join(caffe_root, 'examples/images/cat.jpg'))
image = caffe.io.load_image(os.path.join(root_dir, 'data/dog.jpg'))
# image = caffe.io.load_image(os.path.join(root_dir, 'data/whistle.jpg'))
transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()


labels_file = os.path.join(root_dir, 'data/ilsvrc12/synset_words.txt')
labels = np.loadtxt(labels_file, str, delimiter='\t')

print '=== output label ==='
print labels[output_prob.argmax()]
