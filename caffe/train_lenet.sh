#!/usr/bin/env sh
set -e

caffe train --solver=mnist/lenet_solver.prototxt $@
