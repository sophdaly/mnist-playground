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
BATCH_SIZE = 100


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
    logits, _ = conv_mnist.inference(images_pl, keep_prob_pl, FLAGS.summaries)

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
                # Since Tf variables are only alive inside a session you
                # have to save the model inside the session
                checkpoint_file = utils.git_hash_file_path(FLAGS.model_dir,
                                                           'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=i)
                # This will save the following files:
                # 1) Meta Graph
                # .meta file:
                # - MetaGraphDef protocol buffer representation of MetaGraph
                # which saves the complete Tf Graph structure i.e. the
                # GraphDef that describes the dataflow and all metadata
                # associated with it i.e. all variables, operations,
                # collections, etc. necessary to restore training or inference
                # processes
                # - importing the graph structure will recreate the Graph and
                # all its variables, then the corresponding values for these
                # variables can be restored from the checkpoint file as shown
                # in restore_graph.py
                # - you can reconstruct all of the information in the
                # MetaGraphDef by re-executing the Python code that builds the
                # model n.b. you must recreate the EXACT SAME variables first
                # before restoring their values from the checkpoint as shown
                # in restore_variables.py
                # - since Meta Graph file is not always needed, switch off
                # writing the file in saver.save using write_meta_graph=False

                # 2) Checkpoint files
                # .data file:
                # - binary file containing VALUES of all saved variables
                # outlined in tf.train.Saver() (default is all variables)
                # .index file:
                # - immutable table describing all tensors and their metadata
                # checkpoint file:
                # - keeps a record of latest checkpoint files saved

                # Evaluate training data
                print("Evaluating Training Data:")
                conv_mnist.evaluate(sess, mnist.train, accuracy_op, images_pl,
                                    labels_pl, keep_prob_pl, 0.1, BATCH_SIZE)

                # Evaluate test data
                print("Evaluating Test Data:")
                conv_mnist.evaluate(sess, mnist.test, accuracy_op, images_pl,
                                    labels_pl, keep_prob_pl, 1, BATCH_SIZE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Path to input data directory')
    parser.add_argument('--summaries', action='store_true',
                        help='Generate Tensorboard summaries')
    parser.add_argument('--log_dir', type=str,
                        default='logs/mnist',
                        help='Path to log directory')
    parser.add_argument('--model_dir', type=str,
                        default='models/mnist',
                        help='Path to model directory')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of steps to run trainer')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
