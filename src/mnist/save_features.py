"""
Train ConvNet model and save features from fully connected layer to use as
input for classical machine learning approaches
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
BATCH_SIZE = 100


def main(_):
    # Read data
    mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

    # Genererate placeholders
    images_pl = tf.placeholder(tf.float32, shape=[None, 784], name='images_pl')
    labels_pl = tf.placeholder(tf.float32, shape=[None, NO_CLASSES],
                               name='labels_pl')
    keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob_pl')

    # Add inference, loss, train and accuracy ops to Graph
    logits, _ = conv_mnist.inference(images_pl, keep_prob_pl, FLAGS.summaries)
    loss_op = conv_mnist.loss(logits, labels_pl)
    train_op = conv_mnist.train(loss_op, LEARNING_RATE)
    accuracy_op = conv_mnist.accuracy(logits, labels_pl)

    with tf.Session() as sess:
        # Build summary tensor
        summary_op = tf.summary.merge_all()
        summary_file = utils.git_hash_file_path(FLAGS.log_dir, 'summary')
        summary_writer = tf.summary.FileWriter(summary_file, sess.graph)

        init = tf.global_variables_initializer()

        # Get trainable variables to restore
        variables_to_restore = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)

        # Add ops to save variables to checkpoints
        saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=3)

        sess.run(init)

        # Training cycle
        for i in range(FLAGS.steps):
            t0 = time.time()

            feed_dict = conv_mnist.fill_feed_dict(mnist.train, images_pl,
                                                  labels_pl, keep_prob_pl,
                                                  0.5, BATCH_SIZE)

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
                checkpoint_file = utils.git_hash_file_path(FLAGS.model_dir,
                                                           'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=i)

                # Evaluate test data
                print("Evaluating Test Data:")
                conv_mnist.evaluate(sess, mnist.test, accuracy_op, images_pl,
                                    labels_pl, keep_prob_pl, 1, BATCH_SIZE)


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
