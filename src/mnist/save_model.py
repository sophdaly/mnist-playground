"""
Trains, evaluates and saves MNIST ConvNet model
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import conv_mnist
import utils
import argparse
import time
import sys
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
NO_CLASSES = 10
LEARNING_RATE = 1e-4


def main(_):
    # Load MNIST data into memory
    mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

    # Generate placeholder variables to represent input tensors
    # Note: add names to tensors that need to be restored
    images_pl = tf.placeholder(tf.float32, shape=[None, 784], name='images_pl')
    labels_pl = tf.placeholder(tf.float32, shape=[None, NO_CLASSES],
                               name='labels_pl')
    keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob_pl')

    # Build Graph that computes predictions from inference model
    logits = conv_mnist.inference(images_pl, keep_prob_pl, FLAGS.summaries)

    # Add loss operation to Graph
    loss_op = conv_mnist.loss(logits, labels_pl)

    # Add training operation to Graph
    train_op = conv_mnist.train(loss_op, LEARNING_RATE)

    # Add accuracy operation to Graph
    accuracy_op = conv_mnist.accuracy(logits, labels_pl)

    with tf.Session() as sess:
        # Build summary Tensor based on collection of Summaries
        summary_op = tf.summary.merge_all()

        # Instantiate summary writer for training
        summary_file = utils.git_hash_file_path(FLAGS.log_dir, 'summary')
        summary_writer = tf.summary.FileWriter(summary_file, sess.graph)

        # Add variable initializer
        init = tf.global_variables_initializer()

        # Get trainable variables to restore
        # Note: you can create custom GraphKey collections and add specific
        # variables to it to restore
        variables_to_restore = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)

        # Add ops to save variables to checkpoints
        # Unless specified Saver will save ALL named variables in Graph
        # Maximum of 3 latest models are sasved
        saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=3)

        # Run variable initializer
        sess.run(init)

        # Training cycle
        for i in range(FLAGS.steps):
            t0 = time.time()

            # Fill feed dict with training data and dropout keep prob
            feed_dict = conv_mnist.fill_feed_dict(mnist.train, images_pl,
                                                  labels_pl, keep_prob_pl, 0.5)

            # Train model
            sess.run(train_op, feed_dict=feed_dict)

            # Write summaries and log status
            if i % 100 == 0:
                summary, acc, los = sess.run(
                    [summary_op, accuracy_op, loss_op],
                    feed_dict=feed_dict)
                summary_writer.add_summary(summary, i)
                print(("Step: {}, Loss: {}, Accuracy: {}, [timer: {:.2f}s]")
                      .format(i, los, acc, time.time() - t0))

            # Save checkpoints and evaluate model periodically
            if (i + 1) % 1000 == 0 or (i + 1) == FLAGS.steps:
                # Since Tf variables are only alive inside a session you
                # have to save the model inside the session
                checkpoint_file = utils.git_hash_file_path(FLAGS.model_dir,
                                                           'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=i)

                # Evaluate training data
                print("Evaluating Training Data:")
                conv_mnist.evaluate(sess, mnist.train, accuracy_op, images_pl,
                                    labels_pl, keep_prob_pl, 0.1)

                # Evaluate test data
                print("Evaluating Test Data:")
                conv_mnist.evaluate(sess, mnist.test, accuracy_op, images_pl,
                                    labels_pl, keep_prob_pl, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist_data',
                        help='Path to input data directory')
    parser.add_argument('--summaries', action='store_true',
                        help='Generate Tensorboard summaries')
    parser.add_argument('--log_dir', type=str,
                        default='logs/conv_mnist',
                        help='Path to log directory')
    parser.add_argument('--model_dir', type=str,
                        default='models/conv_mnist',
                        help='Path to model directory')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of steps to run trainer')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
