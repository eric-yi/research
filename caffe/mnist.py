#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pylab import *
import caffe
from caffe import layers as L, params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto()
    
# with open('mnist/lenet_auto_train.prototxt', 'w') as f:
#     f.write(str(lenet('mnist/mnist_train_lmdb', 64)))
    
# with open('mnist/lenet_auto_test.prototxt', 'w') as f:
#     f.write(str(lenet('mnist/mnist_test_lmdb', 100)))

solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')
print '=== net blobs ==='
for k, v in solver.net.blobs.items():
    print k, v.data.shape

print '=== net params ==='
for k, v in solver.net.params.items():
    print k, v[0].data.shape


solver.net.forward()  # train net
r = solver.test_nets[0].forward()  # test net (there can be more than one)
print '=== forward result ==='
print r

print '=== train labels ==='
print solver.net.blobs['label'].data[:8]

print '=== test labels ==='
print solver.test_nets[0].blobs['label'].data[:8]


print '=== training and testing ==='
niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

print '=== train loss ==='
print train_loss
print '=== test acc ==='
print test_acc
