"""
Restores variable values from previously trained MNIST ConvNet model, re runs
inference process and evaluates accuracy against test data
"""

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import conv_mnist
import utils
import argparse
import sys
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
NO_CLASSES = 10
BATCH_SIZE = 100


def forward_pass():
    '''
    Compute forward pass of images through whole network
    '''
    images_pl = tf.placeholder(tf.float32, shape=[None, 784], name='images_pl')
    keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob_pl')

    # Build graph that computes predictions from inference model
    # Set summaries flag to False
    logits, _ = conv_mnist.inference(images_pl, keep_prob_pl, False)
    return logits, images_pl, keep_prob_pl


def evaluate_model(logits, images_pl, keep_prob_pl, sess, data):
    '''
    Evaluate accuracy of model against data
    '''
    labels_pl = tf.placeholder(tf.float32, shape=[None, NO_CLASSES],
                               name='labels_pl')
    accuracy_op = conv_mnist.accuracy(logits, labels_pl)

    print("Evaluating Test Data via Running Restored Inference Process:")
    conv_mnist.evaluate(sess, data, accuracy_op, images_pl, labels_pl,
                        keep_prob_pl, 1, BATCH_SIZE)


def main(_):
    mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

    # Get latest checkpoint file from dir
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)

    # Compute forward pass
    logits, images_pl, keep_prob_pl = forward_pass()

    # Add ops to restore values of the variables created from forward pass
    # from checkpoints
    saver = tf.train.Saver(tf.global_variables())

    # Start session
    with tf.Session() as sess:

        # Restore previously trained variables from disk
        print("Restoring Saved Variables from Checkpoint: {}"
              .format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

        # Optionally print restored operations, variables and saved values
        if FLAGS.print_data:
            print("Restored Variables from MetaGraph:")
            for var in tf.global_variables():
                print(var.name)
            print("Restored Saved Variables from Checkpoint:")
            for init_var in tf.global_variables():
                try:
                    print("{}: {}".format(init_var.name, init_var.eval()))
                except Exception:
                    pass

        evaluate_model(logits, images_pl, keep_prob_pl, sess, mnist.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist_data',
                        help='Path to input data directory')
    parser.add_argument('--model_dir', type=str,
                        default=utils.git_hash_dir_path('models/conv_mnist'),
                        help='Path to model directory')
    parser.add_argument('--print_data', action='store_true',
                        help='Print restored data')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
