"""
Plot 2D embedding
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
import numpy as np
import random
import argparse
import os

FLAGS = None
NO_CLASSES = 10


def load_features(file):
    d = np.load(file)
    images = d['images']
    labels = d['labels']
    feats = d['embedded_features']
    return images, labels, feats


def plot_embedding(X, Y, X_embedded, min_dist=100.0):
    plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.title("t-SNE Embedding of MNIST")
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0,
                        hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, marker="x")
    shown_images = np.array([[1., 1.]])
    indices = np.arange(X_embedded.shape[0])
    random.shuffle(indices)
    for i in indices[:100]:
        dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
        if np.min(dist) < min_dist:
            continue
        shown_images = np.r_[shown_images, [X_embedded[i]]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X[i].reshape(28, 28),
                                      cmap=pl.cm.gray_r), X_embedded[i])
        ax.add_artist(imagebox)


def main():
    train_data = os.path.join(FLAGS.model_dir, "embedded_train_feats.npz")
    test_data = os.path.join(FLAGS.model_dir, "embedded_test_feats.npz")

    # Load features
    train_images, train_labels, train_feats = load_features(train_data)
    test_images, test_labels, test_feats = load_features(test_data)

    # Concatenate training and test data
    feats = np.concatenate((train_feats, test_feats))
    labels = np.concatenate((train_labels, test_labels))
    images = np.concatenate((train_images, test_images))

    plot_embedding(images, labels, feats)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to features directory')

    FLAGS, unparsed = parser.parse_known_args()
    main()
