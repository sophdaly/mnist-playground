'''
Visualises embeddings for mnist
'''

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import numpy as np
import argparse
import sprite
import utils
import sys
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
IMAGE_SIZE = 28
BATCH_SIZE = 5000


def create_mnist_metadata(labels):
    '''
    Returns tsv metadata file associating label to each image
    Index: index in our embedding matrix
    Label: label of the mnist image
    '''
    metadata_file = open(os.path.join(FLAGS.log_dir, 'metadata.tsv'), 'w')
    metadata_file.write('Index\tLabel\n')
    for i in range(BATCH_SIZE):
        metadata_file.write("{}\t{}\n".format(i, labels[i]))
    metadata_file.close()


def create_znist_metadata(labels):
    '''
    Returns tsv metadata file associating label to each image
    Index: index in our embedding matrix
    Label: label of the znist image
    '''
    names = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
             'shirt', 'sneaker', 'bag', 'ankle_boots']
    metadata_file = open(os.path.join(FLAGS.log_dir, 'metadata.tsv'), 'w')
    metadata_file.write('Index\tLabel\n')
    named_labels = np.array([names[j] for j in labels])
    for i in range(BATCH_SIZE):
        metadata_file.write("{}\t{}\n".format(i, named_labels[i]))
    metadata_file.close()


def main(_):
    mnist = read_data_sets(FLAGS.data_dir)
    images, labels = mnist.train.next_batch(BATCH_SIZE)

    # Create embedding
    embedding_var = tf.Variable(images, name="embedding")

    # Instantiate summary writer
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir)

    # Create embedding projector
    config = projector.ProjectorConfig()

    # Add embedding
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify metadata file
    if FLAGS.znist:
        create_znist_metadata(labels)
    else:
        create_mnist_metadata(labels)
    embedding.metadata_path = 'metadata.tsv'

    # Specify sprite file
    if FLAGS.create_sprite:
        sprite.create_sprite(images, FLAGS.log_dir)
    embedding.sprite.image_path = 'sprite.png'

    # Specify dimensions of each thumbnail
    embedding.sprite.single_image_dim.extend([IMAGE_SIZE, IMAGE_SIZE])

    # Visualise embeddings by writing a projector_config.pbtxt for Tensorboard
    # to read on startup
    projector.visualize_embeddings(summary_writer, config)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file)

    print("Run 'tensorboard --logdir \"{}\"',".format(FLAGS.log_dir))
    print("Open http://localhost:6006")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist_data',
                        help='Path to input data directory')
    parser.add_argument('--log_dir', type=str,
                        default=utils.git_hash_dir_path('logs/conv_mnist/'),
                        help='Path to log directory')
    parser.add_argument('--create_sprite', action='store_true',
                        help='Create sprite image')
    parser.add_argument('--znist', action='store_true',
                        help='Create znist metadata file')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
