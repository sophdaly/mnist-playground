"""
Simple MNIST classifier using Softmax regression
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import argparse
import sys
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
LEARNING_RATE = 0.5
BATCH_SIZE = 50
DISPLAY_STEP = 100


def main(_):

    # Load MNIST data into memory
    mnist = read_data_sets('../' + FLAGS.data, one_hot=True)

    # Start TF session
    sess = tf.InteractiveSession()

    # Create model
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Implement regression model for Softmax classifier
    y_pred = tf.nn.bias_add(tf.matmul(x, W), b)

    # Cross entropy loss fn
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))

    # Training step
    # Minimise cross entropy using gradient descent
    # Adding operations to computational graph for computing gradients,
    # computing parameter update steps, and applying update steps to parameters
    train_step = tf.train.GradientDescentOptimizer(
        LEARNING_RATE).minimize(cross_entropy)

    # Use session to execute filling data into tensor variables
    sess.run(tf.global_variables_initializer())

    # Evaluation fns
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Training cycle
    for i in range(1000):
        # Get batch of images + labels
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

        # Perform forward + backward pass with current batch
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

        # Print accuracy and loss for test set
        if i % DISPLAY_STEP == 0:
            acc = accuracy.eval(
                feed_dict={x: mnist.test.images, y_: mnist.test.labels}
            )
            loss = cross_entropy.eval(
                feed_dict={x: mnist.test.images, y_: mnist.test.labels}
            )

            print("Cross Entropy: {}, Accuracy: {}".format(loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/mnist_data',
                        help='Path to input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
