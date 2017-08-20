"""
Restores previously trained MNIST ConvNet model and evaluates against
test data
"""

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import conv_mnist
import argparse
import sys
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
NO_CLASSES = 10


def main(_):
    mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

    # Load graph
    meta_graph = FLAGS.checkpoint_dir + '.meta'
    saver = tf.train.import_meta_graph(meta_graph)

    with tf.Session() as sess:
        # Restore previously trained variables from checkpoint
        print("Restoring Model: {}".format(FLAGS.checkpoint_dir))
        saver.restore(sess, FLAGS.checkpoint_dir)

        # Retrieve protobuf graph definition
        graph = tf.get_default_graph()

        # List operations available in the graph
        print("Restored Operations:")
        for op in graph.get_operations():
            print(op.name)

        # Access saved placeholder variables to feed new data
        images_pl = graph.get_tensor_by_name("images_pl:0")
        labels_pl = graph.get_tensor_by_name("labels_pl:0")
        keep_prob_pl = graph.get_tensor_by_name("keep_prob_pl:0")

        # Access saved operation to re run
        accuracy_op = graph.get_tensor_by_name("accuracy_op:0")

        print("Evaluating Test Data:")
        conv_mnist.evaluate(sess, mnist.test, images_pl, labels_pl,
                            keep_prob_pl, 0.1, accuracy_op)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist_data',
                        help='Path to input data directory')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='logs/conv_mnist/model/model.ckpt-4999',
                        help='Path to checkpoint directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
