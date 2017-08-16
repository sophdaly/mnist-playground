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


def variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + '/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name + '/stddev', stddev)


def activation_summaries(x, name):
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.histogram(name + '/activations', x)
    variable_summaries(x, name)


def conv2d_layer(layer_name, input_tensor, filter_size, no_filters,
                 act=tf.nn.relu, summaries=False):
    '''
    Implement 2D convolution, non-linearize with ReLU
    '''
    with tf.name_scope(layer_name):
        input_dim = input_tensor.get_shape()[-1].value

        with tf.variable_scope(layer_name):
            weight = weight_variable([filter_size, filter_size,
                                     input_dim, no_filters])
            bias = bias_variable([no_filters])
            if summaries:
                variable_summaries(weight, layer_name + '/weight')
                variable_summaries(bias, layer_name + '/bias')

        with tf.name_scope('activations'):
            preactivate = tf.nn.bias_add(conv2d(input_tensor, weight), bias)
            activation = act(preactivate, "activation")
            if summaries:
                activation_summaries(activation, layer_name)

            return activation


def max_pool2d_layer(layer_name, input_tensor, pool_size, stride):
    '''
    Apply 2x2 max pooling
    '''
    with tf.name_scope(layer_name):
        pooled = tf.nn.max_pool(
            input_tensor, ksize=[1, pool_size, pool_size, 1],
            strides=[1, stride, stride, 1], padding='SAME'
        )
        return pooled


def fc_layer(layer_name, input_tensor, no_filters, act=tf.nn.relu,
             dropout=None, summaries=False):
    '''
    Flatten input_tensor if needed, non-linearize with ReLU and optionally
    apply dropout
    '''
    with tf.name_scope(layer_name):
        # Reshape input tensor to flatten tensor if needed
        input_shape = input_tensor.get_shape()
        if len(input_shape) == 4:
            input_dim = np.int(np.product(input_shape[1:]))
            input_tensor = tf.reshape(input_tensor, [-1, input_dim])
        elif len(input_shape) == 2:
            input_dim = input_shape[-1].value
        else:
            raise RuntimeError('Input Tensor Shape: {}'.format(input_shape))

        with tf.variable_scope(layer_name):
            weight = weight_variable([input_dim, no_filters])
            bias = bias_variable([no_filters])
            if summaries:
                variable_summaries(weight, layer_name + '/weight')
                variable_summaries(bias, layer_name + '/bias')

        with tf.name_scope('activations'):
            preactivate = tf.nn.bias_add(tf.matmul(input_tensor, weight), bias)
            activation = act(preactivate, "activation")
            if dropout is not None:
                activation = tf.nn.dropout(activation, dropout)
            if summaries:
                activation_summaries(activation, layer_name)

            return activation


def inference(x, keep_prob, summaries):
    '''
    Build network graph to compute output predictions for images
    '''
    # Reshape x to 4d tensor
    # [BATCH_SIZE, 784] -> [BATCH_SIZE, width, height, no_color_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    out = conv2d_layer(input_tensor=x_image, filter_size=5, no_filters=32,
                       layer_name='conv_1', summaries=summaries)
    out = max_pool2d_layer(input_tensor=out, pool_size=2, stride=2,
                           layer_name='max_pool_1')
    out = conv2d_layer(input_tensor=out, filter_size=5, no_filters=64,
                       layer_name='conv_2', summaries=summaries)
    out = max_pool2d_layer(input_tensor=out, pool_size=2, stride=2,
                           layer_name='max_pool_2')
    out = fc_layer(input_tensor=out, no_filters=1024, dropout=keep_prob,
                   layer_name='fc_1', summaries=summaries)
    logits = fc_layer(input_tensor=out, no_filters=10, act=tf.identity,
                      layer_name='fc_2', summaries=summaries)

    return logits


def loss(logits, labels):
    '''
    Calculate cross entropy loss
    '''
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy)


def train(loss, learning_rate):
    '''
    Train model by optimizing gradient descent
    Add summary to track loss on TensorBoard
    '''
    # Create gradient descent optimizer with learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    tf.summary.scalar('loss', loss)
    return train_op


def evaluate(logits, labels):
    '''
    Evaluate accuracy of logits at predicting labels
    Add summary to track accuracy on TensorBoard
    '''
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy_op)
    return accuracy_op


def main(_):
    # Load MNIST data into memory
    mnist = read_data_sets('../' + FLAGS.data, one_hot=True)

    # Start session
    sess = tf.InteractiveSession()

    # Generate placeholder variables to represent input tensors
    images = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Build Graph that computes predictions from inference model
    logits = inference(images, keep_prob, FLAGS.tensorboard)

    # Add loss operation to Graph
    loss_op = loss(logits, labels)

    # Add training operation to Graph
    train_op = train(loss_op, LEARNING_RATE)

    # Add accuracy operation to Graph
    accuracy_op = evaluate(logits, labels)

    with tf.Session() as sess:
        # Build summary Tensor based on collection of Summaries
        summary_op = tf.summary.merge_all()
        # Instantiate summary writer for training
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             sess.graph)
        # Instantiate summary writer for testing
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test',
                                            sess.graph)

        # Add variable initializer
        init = tf.global_variables_initializer()
        sess.run(init)

        # Number of images in test data
        no_vals = mnist.validation.labels.shape[0]

        # Training cycle
        t0 = time.time()
        for i in range(5000):
            # Train and record train set summaries
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _ = sess.run([summary_op, train_op], feed_dict={
                                  images: batch_xs,
                                  labels: batch_ys,
                                  keep_prob: 0.5
                                  })
            train_writer.add_summary(summary, i)

            if i % DISPLAY_STEP == 0:
                # Evaluate accuracy on test set and record summaries
                # Subsample test set (for speed)
                mean_batch_time = (time.time() - t0) / 100.0
                t1 = time.time()
                sample = np.random.choice(range(no_vals), 500, replace=False)

                summary, acc, los = sess.run(
                    [summary_op, accuracy_op, loss_op],
                    feed_dict={images: mnist.validation.images[sample],
                               labels: mnist.validation.labels[sample],
                               keep_prob: 1.0}
                )
                test_writer.add_summary(summary, i)

                print(("Iter: {}, Loss: {}, Accuracy: {} "
                      "[train batch time: {:.2f}s, eval time: {:.2f}s]")
                      .format(i, los, acc, mean_batch_time, time.time() - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist_data',
                        help='Path to input data directory')
    parser.add_argument('--tensorboard', action='store_true',
                        help='TensorBoard on/off ')
    parser.add_argument('--log_dir', type=str,
                        default='logs/mnist/conv_mnist',
                        help='Path to log directory')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
