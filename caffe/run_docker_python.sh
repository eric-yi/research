#!/bin/bash

docker run -it --rm -e DISPLAY=docker.for.mac.host.internal:0 -v "$(pwd)":/workspace elezar/caffe:cpu python $@
