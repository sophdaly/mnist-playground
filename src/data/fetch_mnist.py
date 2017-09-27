"""
Download MNIST data
"""

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

read_data_sets('data/mnist', one_hot=True)
