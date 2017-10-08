"""
KNN classification using feature space extracted from trained ConvNet model
and embedded to 2 dimensions with t-SNE
"""

from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import argparse
import utils
import os

FLAGS = None
K = 50


def load_features(file):
    d = np.load(file)
    labels = d['labels']
    feats = d['embedded_features']
    return feats, labels


def main():
    train_data = os.path.join(FLAGS.model_dir, "embedded_train_feats.npz")
    test_data = os.path.join(FLAGS.model_dir, "embedded_test_feats.npz")

    # Load features
    train_feats, train_labels = load_features(train_data)
    test_feats, test_labels = load_features(test_data)

    knn = KNeighborsClassifier(n_neighbors=K)
    knn_model = knn.fit(train_feats, train_labels)

    knn_test_preds = knn_model.predict(test_feats)

    # Calculate accuracy
    knn_accuracy = np.sum(knn_test_preds == test_labels) / 100.0
    print("Accuracy of KNN in t-SNE Space: {}%".format(knn_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default=utils.git_hash_dir_path('models/conv_mnist'),
                        help='Path to features directory')
    parser.add_argument('--k', type=int, default=0,
                        help='Number of nearest neighbours')

    FLAGS, unparsed = parser.parse_known_args()
    main()
