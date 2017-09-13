"""
Extract features from trained ConvNet model and save to numpy arrays
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import conv_mnist
import utils
import argparse
import numpy as np
import sys
import os

FLAGS = None
NO_CLASSES = 10
NO_FILTERS = 1024
BATCH_SIZE = 100


def extract_features(sess, data, file_name, images_pl, labels_pl, keep_prob_pl,
                     prediction_tensor, representation_tensor):
    '''
    Run prediction and respresentatin tensors restored from trained ConvNet
    model, extract predictions and features and save to numpy array
    '''
    # Create numpy arrays
    features = np.zeros((0, NO_FILTERS), dtype='float32')
    predictions = np.zeros((0), dtype='float32')
    labels = np.zeros((0), dtype='float32')

    epoch = data.num_examples // BATCH_SIZE
    for i in range(epoch):

        batch_images, batch_labels = data.next_batch(BATCH_SIZE)
        feed_dict = {images_pl: batch_images,
                     labels_pl: batch_labels,
                     keep_prob_pl: 1}

        # Evaluate tensors
        [preds, feats] = sess.run([prediction_tensor, representation_tensor],
                                  feed_dict=feed_dict)

        # Concatenate tensor values to arrays
        # Decode one hot for labels and predictions
        features = np.concatenate((features, feats))
        predictions = np.concatenate((predictions, preds.argmax(1)))
        labels = np.concatenate((labels, batch_labels.argmax(1)))

    # Save arrays to compressed .npz file
    np.savez_compressed(file_name, features=features, predictions=predictions,
                        labels=labels)


def main(_):
    data = read_data_sets(FLAGS.data_dir, one_hot=True)

    # Get latest checkpoint file from dir
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)

    # Generate necessary placeholders
    images_pl = tf.placeholder(tf.float32, shape=[None, 784], name='images_pl')
    labels_pl = tf.placeholder(tf.float32, shape=[None, NO_CLASSES],
                               name='labels_pl')
    keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob_pl')

    # Add inference op to Graph to extract predictions and features
    prediction_tensor, representation_tensor = conv_mnist.inference(
        images_pl, keep_prob_pl, False)

    # Add ops to restore values of the variables created from checkpoints
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:

        # Restore previously trained variables from disk
        print("Restoring saved variables from checkpoint: {}"
              .format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

        # Extract features from training data
        train_file = os.path.join(FLAGS.model_dir, "train_features.npz")
        print("Extracting training data features to: {}".format(train_file))
        extract_features(sess, data.train, train_file, images_pl,
                         labels_pl, keep_prob_pl, prediction_tensor,
                         representation_tensor)

        # Extract features from test data
        test_file = os.path.join(FLAGS.model_dir, "test_features.npz")
        print("Extracting test data features to: {}".format(test_file))
        extract_features(sess, data.test, test_file, images_pl,
                         labels_pl, keep_prob_pl, prediction_tensor,
                         representation_tensor)

        print("Features extracted to directory: {}".format(FLAGS.model_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist_data',
                        help='Path to checkpoint directory')
    parser.add_argument('--model_dir', type=str,
                        default=utils.git_hash_dir_path('models/conv_mnist'),
                        help='Path to model directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
