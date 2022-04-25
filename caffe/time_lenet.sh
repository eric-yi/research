#!/usr/bin/env sh
set -e

caffe time -model mnist/lenet_train_test.prototxt $@
