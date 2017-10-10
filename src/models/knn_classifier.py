"""
k-Nearest Neighbours classification with cross validation using embedded
feature space extracted from trained ConvNet model
"""

from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import utils
import os
import matplotlib.pyplot as plt


FLAGS = None
K_FOLD = 10


def load_features(file):
    d = np.load(file)
    labels = d['labels']
    feats = d['embedded_features']
    return feats, labels


def cross_validate(k_fold, X, Y, plot=False):
    '''
    Perform k-fold cross validation
    Return value of K nearest neighbours which achieves highest accuracy
    '''

    # Create list of odd K values
    neighbours = filter(lambda x: x % 2 != 0, list(range(1, 50)))

    # Create empty list to hold cross validation scores
    cv_scores = []

    # Perform k-fold cross validation
    for k in neighbours:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, Y, cv=k_fold, scoring='accuracy')
        cv_scores.append(scores.mean())

    max_score = max(cv_scores)

    k = neighbours[cv_scores.index(max_score)]
    print("Maximum CV accuracy of {} achieved at K = {}".format(max_score, k))

    if plot:
        plt.plot(neighbours, cv_scores)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validated Accuracy')
        plt.show()

    return k


def main():
    train_data = os.path.join(FLAGS.model_dir, "embedded_train_feats.npz")
    test_data = os.path.join(FLAGS.model_dir, "embedded_test_feats.npz")

    # Load features
    train_feats, train_labels = load_features(train_data)
    test_feats, test_labels = load_features(test_data)

    # Split training data for validation
    train_feats, val_feats, train_labels, val_labels = train_test_split(
        train_feats, train_labels, test_size=0.1, random_state=6
    )

    # Perform cross validation if value for k not provided
    if FLAGS.k is None:
        FLAGS.k = cross_validate(K_FOLD, val_feats, val_labels)

    # Create and fit nearest-neighbour classifier
    knn = KNeighborsClassifier(n_neighbors=FLAGS.k)
    knn_model = knn.fit(train_feats, train_labels)

    # Predict labels for test data
    knn_test_preds = knn_model.predict(test_feats)

    # Calculate accuracy
    knn_accuracy = np.sum(knn_test_preds == test_labels) / 100.0
    print("Accuracy of kNN with {} neighbours in t-SNE space: {}%".format(
        FLAGS.k, knn_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default=utils.git_hash_dir_path('models/conv_mnist'),
                        help='Path to features directory')
    parser.add_argument('--k', type=int, help='Number of nearest neighbours')

    FLAGS, unparsed = parser.parse_known_args()
    main()
