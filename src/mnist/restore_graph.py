"""
Restores entire graph from previously trained MNIST ConvNet model and
evaluates against test data
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


def evaluate_restored_model(data, sess, graph):
    '''
    Evaluate accuracy of restored model against data
    '''
    # Access restored placeholder variables to feed new data
    images_pl = graph.get_tensor_by_name("images_pl:0")
    labels_pl = graph.get_tensor_by_name("labels_pl:0")
    keep_prob_pl = graph.get_tensor_by_name("keep_prob_pl:0")

    # Access restored operation to re run
    accuracy_op = graph.get_tensor_by_name("accuracy_op:0")

    print("Evaluating Test Data via Restored Model:")
    conv_mnist.evaluate(sess, data, accuracy_op, images_pl, labels_pl,
                        keep_prob_pl, 1, BATCH_SIZE)


def main(_):
    mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

    # Get latest checkpoint file from dir
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)

    # Load latest checkpoint Graph via import_meta_graph:
    #   - construct protocol buffer from file content
    #   - add all nodes to current graph and recreate collections
    #   - return Saver
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')

    # Start session
    with tf.Session() as sess:

        # Restore previously trained variables from disk
        print("Restoring Model: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

        # Retrieve protobuf graph definition
        graph = tf.get_default_graph()

        # Optionally print restored operations, variables and saved values
        if FLAGS.print_data:
            print("Restored Operations from MetaGraph:")
            for op in graph.get_operations():
                print(op.name)
            print("Restored Variables from MetaGraph:")
            for var in tf.global_variables():
                print(var.name)
            print("Restored Saved Variables from Checkpoint:")
            for init_var in tf.global_variables():
                try:
                    print("{}: {}".format(init_var.name, init_var.eval()))
                except Exception:
                    pass

        evaluate_restored_model(mnist.test, sess, graph)


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
