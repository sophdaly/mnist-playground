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
BATCH_SIZE = 100
NO_CLASSES = 10
IMAGE_SIZE = 28


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


def inference(images, keep_prob, summaries):
    '''
    Build network graph to compute output predictions for images
    '''
    # Reshape images to 4d tensor
    # [BATCH_SIZE, 784] -> [BATCH_SIZE, width, height, no_color_channels]
    input = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    out = conv2d_layer(input_tensor=input, filter_size=5, no_filters=32,
                       layer_name='conv_1', summaries=summaries)
    out = max_pool2d_layer(input_tensor=out, pool_size=2, stride=2,
                           layer_name='max_pool_1')
    out = conv2d_layer(input_tensor=out, filter_size=5, no_filters=64,
                       layer_name='conv_2', summaries=summaries)
    out = max_pool2d_layer(input_tensor=out, pool_size=2, stride=2,
                           layer_name='max_pool_2')
    out = fc_layer(input_tensor=out, no_filters=1024, dropout=keep_prob,
                   layer_name='fc_1', summaries=summaries)
    logits = fc_layer(input_tensor=out, no_filters=NO_CLASSES, act=tf.identity,
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


def accuracy(logits, labels):
    '''
    Calculate accuracy of logits at predicting labels
    Add summary to track accuracy on TensorBoard
    '''
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy_op)
    return accuracy_op


def fill_feed_dict(data, images_pl, labels_pl, keep_prob_pl, keep_prob):
    '''
    Fill feed_dict for training step
    '''
    batch_images, batch_labels = data.next_batch(BATCH_SIZE)
    return {images_pl: batch_images,
            labels_pl: batch_labels,
            keep_prob_pl: keep_prob}


def evaluate(sess, data, images_pl, labels_pl, keep_prob_pl, keep_prob,
             accuracy_op):
    '''
    Evaluate model against data
    '''
    epoch = data.num_examples // BATCH_SIZE
    acc = 0.0
    t0 = time.time()
    for i in range(epoch):
        feed_dict = fill_feed_dict(data, images_pl, labels_pl,
                                   keep_prob_pl, keep_prob)
        acc += sess.run(accuracy_op, feed_dict=feed_dict)

    print(("Epoch Accuracy: {:.4f} [timer: {:.2f}s]")
          .format(acc / epoch, time.time() - t0))


def main(_):
    # Load MNIST data into memory
    mnist = read_data_sets('../' + FLAGS.data_dir, one_hot=True)

    # Start session
    sess = tf.InteractiveSession()

    # Generate placeholder variables to represent input tensors
    images_pl = tf.placeholder(tf.float32, shape=[None, 784])
    labels_pl = tf.placeholder(tf.float32, shape=[None, NO_CLASSES])
    keep_prob_pl = tf.placeholder(tf.float32)

    # Build Graph that computes predictions from inference model
    logits = inference(images_pl, keep_prob_pl, FLAGS.tensorboard)

    # Add loss operation to Graph
    loss_op = loss(logits, labels_pl)

    # Add training operation to Graph
    train_op = train(loss_op, LEARNING_RATE)

    # Add accuracy operation to Graph
    accuracy_op = accuracy(logits, labels_pl)

    with tf.Session() as sess:
        # Build summary Tensor based on collection of Summaries
        summary_op = tf.summary.merge_all()

        # Instantiate summary writer for training
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/summary',
                                               sess.graph)

        # Add variable initializer
        init = tf.global_variables_initializer()

        # Create saver for writing training checkpoints
        # Unless specified Saver will save all variables
        # Maximum of 3 latest models are saved
        saver = tf.train.Saver(max_to_keep=3)

        # Run variable initializer
        sess.run(init)

        # Training cycle
        for i in range(FLAGS.steps):
            t0 = time.time()

            # Fill feed dict with training data and dropout keep prob
            feed_dict = fill_feed_dict(mnist.train, images_pl,
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
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=i)

                # Evaluate training data
                print("Evaluating Training Data:")
                evaluate(sess, mnist.train, images_pl, labels_pl,
                         keep_prob_pl, 0.1, accuracy_op)

                # Evaluate test data
                print("Evaluating Test Data:")
                evaluate(sess, mnist.test, images_pl, labels_pl,
                         keep_prob_pl, 0.5, accuracy_op)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist_data',
                        help='Path to input data directory')
    parser.add_argument('--tensorboard', action='store_true',
                        help='TensorBoard on/off ')
    parser.add_argument('--log_dir', type=str,
                        default='logs/mnist/conv_mnist',
                        help='Path to log directory')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of steps to run trainer')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
