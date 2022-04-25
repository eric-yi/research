#!/usr/bin/env sh
set -e

caffe test -model mnist/lenet_train_test.prototxt -weights mnist/lenet_iter_10000.caffemodel $@
