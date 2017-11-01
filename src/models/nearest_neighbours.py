"""
Nearest Neighbour Search functionality via k-d trees using 2 dimensional
embedding of ZNIST convnet feature space:
- given an image, output it's nearest neighbours
"""

from __future__ import print_function
from sklearn.neighbors import KDTree
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import os

IMAGE_SIZE = 28


def load_features(file):
    d = np.load(file)
    images = d['images']
    labels = d['labels']
    feats = d['embedded_features']
    return images, labels, feats


def find_index(images, image):
    '''
    Hacky way to find index of an image
    '''
    return np.where(np.all(images == image, axis=1))[0][0]


def plot_image(image, label):
    image = np.reshape(image, (IMAGE_SIZE, IMAGE_SIZE))
    plt.title('Label: {label}'.format(label=label))
    plt.imshow(image, cmap=pl.cm.gray_r)
    plt.axis('off')
    plt.show()


def plot_gif(images, labels):
    '''
    Plot images in a gif
    '''
    rc('animation', html='html5')

    images = np.reshape(images, (-1, IMAGE_SIZE, IMAGE_SIZE))
    no_plots = images.shape[0]

    fig = plt.figure(figsize=(2, 2))
    plt.axis('off')
    im = plt.imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), cmap=pl.cm.gray_r)

    def init():
        im.set_data(np.zeros((IMAGE_SIZE, IMAGE_SIZE)))
        return im

    def update(i):
        im.set_data(images[i])
        im.autoscale()
        return im

    anim = animation.FuncAnimation(fig, func=update, init_func=init,
                                   frames=no_plots, interval=500, repeat=False)
    return anim


def plot_images(images, labels):
    '''
    Plot list of images in a row
    '''
    images = np.reshape(images, (-1, IMAGE_SIZE, IMAGE_SIZE))
    no_plots = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]

    # Display images in a col
    display = np.ones((height * no_plots, width))
    for i in range(no_plots):
        image = images[i]
        display[i * height:(i + 1) * height, 0:width] = image

    plt.title("Nearest Neighbours: {}".format(no_plots))
    plt.imshow(display, cmap=pl.cm.gray_r)
    plt.axis('off')
    plt.show()


def load_data(model_dir):
    '''
    Load and merge train and test data
    '''
    train_data = os.path.join(model_dir, "embedded_train_feats.npz")
    test_data = os.path.join(model_dir, "embedded_test_feats.npz")

    images1, labels1, feats1 = load_features(train_data)
    images2, labels2, feats2 = load_features(test_data)

    images = np.concatenate((images1, images2))
    labels = np.concatenate((labels1, labels2))
    feats = np.concatenate((feats1, feats2))

    return images, labels, feats


def nearest_neighbour_search(selected_index, images, feats, labels,
                             no_neighbours):

    # k-d tree
    # - binary tree where each node is k dim point
    # - every non-leaf node is a hyperplane dividing the space into 2 half
    # spaces
    # - nn search is done by starting at root and recursively searching tree
    # - not suitable for high dims, rule: number of points in the data >> 2k
    # - building a kd-tree has O(NlogN) time and O(KN) space complexity
    # - nearest neighbor search: close to O(logN)
    # - M nearest neighbors: close to O(MlogN)

    tree = KDTree(feats)
    dist, inds = tree.query([feats[selected_index]], k=no_neighbours)
    print("Nearest neighbour indices: {}".format(inds[0]))
    return inds
