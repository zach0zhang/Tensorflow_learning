# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:18:43 2018

@author: Administrator
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)

print ("Training data size:", mnist.train.num_examples)