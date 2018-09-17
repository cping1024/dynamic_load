#! /usr/bin/python
# -*- coding=UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import caffe
import os, sys

caffe_root = '/home/allen/sdb1/workspace/open_source/caffe'

os.chdir(caffe_root)
img = caffe.io.load_image('examples/images/cat.jpg')
plt.imshow(img)
plt.axis('off')
