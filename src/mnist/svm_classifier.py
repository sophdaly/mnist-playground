"""
Classify MNIST images using features extracted from trained ConvNet model
"""

from __future__ import print_function
from sklearn import svm
import utils
import argparse
import numpy as np
import os

FLAGS = None


def load_features(file):
    d = np.load(file)
    feats = d['features']
    preds = d['predictions']
    labels = d['labels']
    return feats, preds, labels


def main():
    train_data = os.path.join(FLAGS.model_dir, "train_features.npz")
    test_data = os.path.join(FLAGS.model_dir, "test_features.npz")

    train_feats, train_preds, train_labels = load_features(train_data)
    test_feats, test_preds, test_labels = load_features(test_data)

    # Create SVM classifier
    svc = svm.SVC(kernel='linear', C=0.1)

    # Train
    svc.fit(train_feats, train_labels)

    # Test
    svm_test_preds = svc.predict(test_feats)

    # Calculate accuracy
    svm_accuracy = np.sum(svm_test_preds == test_labels) / 100.0
    cnn_accuracy = np.sum(test_preds == test_labels) / 100.0

    print("Accuracy of SVM: {}%".format(svm_accuracy))
    print("Accuracy of CNN: {}%".format(cnn_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist_data',
                        help='Path to checkpoint directory')
    parser.add_argument('--model_dir', type=str,
                        default=utils.git_hash_dir_path('models/conv_mnist'),
                        help='Path to features directory')

    FLAGS, unparsed = parser.parse_known_args()
    main()
