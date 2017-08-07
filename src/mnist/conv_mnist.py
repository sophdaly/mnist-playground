"""
Deep MNIST classifier using convolutional layers

Input -> Conv -> Pool -> Conv -> Pool -> FullyConnected -> Dropout -> Softmax
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import argparse
import time
import sys
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
DISPLAY_STEP = 100


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv2d_layer(input_tensor, filter_size, no_filters, act=tf.nn.relu):
    '''
    Implement 2D convolution, non-linearize with ReLU and apply 2x2 max pooling
    '''
    input_dim = input_tensor.get_shape()[-1].value
    weight = weight_variable([filter_size, filter_size, input_dim,
                              no_filters])
    bias = bias_variable([no_filters])
    activation = act(conv2d(input_tensor, weight) + bias, "activation")
    activation = max_pool_2x2(activation)

    return activation


def fc_layer(input_tensor, no_filters, act=tf.nn.relu, dropout=None):
    '''
    Flatten input_tensor if needed, non-linearize with ReLU and optionally
    apply dropout
    '''
    # Reshape input tensor to flatten tensor if needed
    input_shape = input_tensor.get_shape()
    if len(input_shape) == 4:
        input_dim = np.int(np.product(input_shape[1:]))
        input_tensor = tf.reshape(input_tensor, [-1, input_dim])
    elif len(input_shape) == 2:
        input_dim = input_shape[-1].value
    else:
        raise RuntimeError('ERROR Input Tensor Shape: {}'.format(input_shape))

    weight = weight_variable([input_dim, no_filters])
    bias = bias_variable([no_filters])

    preactivate = tf.nn.bias_add(tf.matmul(input_tensor, weight), bias)
    activation = act(preactivate, "activation")

    if dropout is not None:
        activation = tf.nn.dropout(activation, dropout)

    return activation


def deep_nn(x, keep_prob):

    # Reshape x to 4d tensor
    # [BATCH_SIZE, 784] -> [BATCH_SIZE, width, height, no_color_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    out_1 = conv2d_layer(input_tensor=x_image, filter_size=5, no_filters=32)
    out_2 = conv2d_layer(input_tensor=out_1, filter_size=5, no_filters=64)
    out_3 = fc_layer(input_tensor=out_2, no_filters=1024, dropout=keep_prob)
    y_pred = fc_layer(input_tensor=out_3, no_filters=10, act=tf.identity)

    return y_pred


def main(_):
    # Load MNIST data into memory
    mnist = read_data_sets('../' + FLAGS.data, one_hot=True)

    # Start TF session
    sess = tf.InteractiveSession()

    # Create model
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Build the graph for deep net
    y_pred = deep_nn(x, keep_prob)

    # Loss fn
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y_pred))

    # Train
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # Evaluation fns
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        no_vals = mnist.validation.labels.shape[0]

        # Training cycle
        t0 = time.time()
        for i in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys,
                           keep_prob: 0.5})

            if i % 10 == 0:
                print("Iter: {}".format(i))

            if i % DISPLAY_STEP == 0:
                print("Evaluating...")
                # Subsample eval set (for speed)
                # Compute output, and compute metrics
                mean_batch_time = (time.time() - t0) / 100.0
                t1 = time.time()
                sample = np.random.choice(range(no_vals), 500, replace=False)

                acc = accuracy.eval(feed_dict={
                                    x: mnist.validation.images[sample],
                                    y_: mnist.validation.labels[sample],
                                    keep_prob: 1.0
                                    })
                loss = cross_entropy.eval(feed_dict={
                                          x: mnist.test.images,
                                          y_: mnist.test.labels,
                                          keep_prob: 1.0
                                          })

                print("Iter: {}, Loss: {}, Accuracy: {}, train batch time: \
                    [{:.2f} s], eval_time:[{:.2f} s]".format(
                    i, loss, acc, mean_batch_time, time.time() - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='data/mnist_data',
                        help='Path to input data')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
