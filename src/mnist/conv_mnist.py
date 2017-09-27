"""
Deep MNIST classifier using convolutional layers

Input -> Conv -> Pool -> Conv -> Pool -> FullyConnected -> Dropout -> Softmax
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NO_CLASSES = 10
IMAGE_SIZE = 28


def _weight_variable(shape):
    return tf.get_variable(name='weights',
                           initializer=tf.truncated_normal(shape, stddev=0.1))


def _bias_variable(shape):
    return tf.get_variable(name='biases',
                           initializer=tf.constant(0.1, shape=shape))


def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + '/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name + '/stddev', stddev)


def _activation_summaries(x, name):
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.histogram(name, x)
    _variable_summaries(x, name)


def _conv2d_layer(layer_name, input_tensor, filter_size, no_filters,
                  act=tf.nn.relu, summaries=False):
    '''
    Implement 2D convolution, non-linearize with ReLU
    '''
    # Add variables and operations to scopes to ensure logical grouping of
    # layers Graph
    with tf.variable_scope(layer_name):
        input_dim = input_tensor.get_shape()[-1].value

        weight = _weight_variable([filter_size, filter_size, input_dim,
                                  no_filters])
        bias = _bias_variable([no_filters])
        preactivate = tf.nn.bias_add(_conv2d(input_tensor, weight), bias)
        activation = act(preactivate, "activation")

        if summaries:
            _variable_summaries(weight, '/weights')
            _variable_summaries(bias, '/biases')
            _activation_summaries(activation, 'activations')

        return activation


def _max_pool2d_layer(layer_name, input_tensor, pool_size, stride):
    '''
    Apply 2x2 max pooling
    '''
    with tf.variable_scope(layer_name):
        pooled = tf.nn.max_pool(
            input_tensor, ksize=[1, pool_size, pool_size, 1],
            strides=[1, stride, stride, 1], padding='SAME')
        return pooled


def _fc_layer(layer_name, input_tensor, no_filters, act=tf.nn.relu,
              dropout=None, summaries=False):
    '''
    Flatten input_tensor if needed, non-linearize with ReLU and optionally
    apply dropout
    '''
    with tf.variable_scope(layer_name):
        # Reshape input tensor to flatten tensor if needed
        input_shape = input_tensor.get_shape()
        if len(input_shape) == 4:
            input_dim = np.int(np.product(input_shape[1:]))
            input_tensor = tf.reshape(input_tensor, [-1, input_dim])
        elif len(input_shape) == 2:
            input_dim = input_shape[-1].value
        else:
            raise RuntimeError('Input Tensor Shape: {}'.format(input_shape))

        weight = _weight_variable([input_dim, no_filters])
        bias = _bias_variable([no_filters])
        preactivate = tf.nn.bias_add(tf.matmul(input_tensor, weight), bias)
        activation = act(preactivate, "activation")

        if dropout is not None:
            activation = tf.nn.dropout(activation, dropout)

        if summaries:
            _variable_summaries(weight, '/weights')
            _variable_summaries(bias, '/biases')
            _activation_summaries(activation, '/activations')

        return activation


def inference(images, keep_prob, summaries):
    '''
    Build network graph to compute output predictions for images
    '''
    # Reshape images to 4d tensor
    # [batch, 784] -> [batch, width, height, no_color_channels]
    input = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    out = _conv2d_layer(input_tensor=input, filter_size=5, no_filters=32,
                        layer_name='conv_1', summaries=summaries)
    out = _max_pool2d_layer(input_tensor=out, pool_size=2, stride=2,
                            layer_name='max_pool_1')
    out = _conv2d_layer(input_tensor=out, filter_size=5, no_filters=64,
                        layer_name='conv_2', summaries=summaries)
    out = _max_pool2d_layer(input_tensor=out, pool_size=2, stride=2,
                            layer_name='max_pool_2')
    out = _fc_layer(input_tensor=out, no_filters=1024, dropout=keep_prob,
                    layer_name='fc_1', summaries=summaries)

    # Representation layer
    features = _fc_layer(input_tensor=out, no_filters=1024, dropout=keep_prob,
                         layer_name='fc_1', summaries=summaries)
    # Prediction layer
    logits = _fc_layer(input_tensor=features, no_filters=NO_CLASSES,
                       act=tf.identity, layer_name='fc_2',
                       summaries=summaries)

    return logits, features


def loss(logits, labels):
    '''
    Calculate cross entropy loss
    '''
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy, name='loss_op')


def train(loss, learning_rate):
    '''
    Train model by optimizing gradient descent
    Add summary to track loss on TensorBoard
    '''
    # Create gradient descent optimizer with learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, name='train_op')
    tf.summary.scalar('loss', loss)
    return train_op


def accuracy(logits, labels):
    '''
    Calculate accuracy of logits at predicting labels
    Add summary to track accuracy on TensorBoard
    '''
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                 name='accuracy_op')
    tf.summary.scalar('accuracy', accuracy_op)
    return accuracy_op


def fill_feed_dict(data, images_pl, labels_pl, keep_prob_pl, keep_prob, batch):
    '''
    Fill feed_dict for training step
    '''
    batch_images, batch_labels = data.next_batch(batch)
    return {images_pl: batch_images,
            labels_pl: batch_labels,
            keep_prob_pl: keep_prob}


def evaluate(sess, data, accuracy_op, images_pl, labels_pl, keep_prob_pl,
             keep_prob, batch):
    '''
    Evaluate model against data
    '''
    epoch = data.num_examples // batch
    acc = 0.0
    t0 = time.time()
    for i in range(epoch):
        feed_dict = fill_feed_dict(data, images_pl, labels_pl, keep_prob_pl,
                                   keep_prob, batch)
        acc += sess.run(accuracy_op, feed_dict=feed_dict)

    print(("Epoch Accuracy: {:.4f} [timer: {:.2f}s]")
          .format(acc / epoch, time.time() - t0))
