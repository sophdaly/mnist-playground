"""
Visualise embeddings for mnist
"""

from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import numpy as np
import argparse
import sprite
import sys
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
IMAGE_SIZE = 28
BATCH_SIZE = 5000


def create_mnist_metadata(labels, file):
    '''
    Returns tsv metadata file associating label to each image
    Index: index in our embedding matrix
    Label: label of the mnist image
    '''
    file.write('Index\tLabel\n')
    for i in range(labels.size):
        file.write("{}\t{}\n".format(i, labels[i]))
    file.close()


def create_znist_metadata(labels, file):
    '''
    Returns tsv metadata file associating label to each image
    Index: index in our embedding matrix
    Label: label of the znist image
    '''
    names = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
             'shirt', 'sneaker', 'bag', 'ankle_boots']
    file.write('Index\tLabel\n')
    named_labels = np.array([names[j] for j in labels])
    for i in range(labels.size):
        file.write("{}\t{}\n".format(i, named_labels[i]))
    file.close()


def load_predictions(file):
    d = np.load(file)
    images = d['images']
    svm_preds = d['svm_preds']
    return images, svm_preds


def main(_):

    # Load predictions
    pred_file = os.path.join(FLAGS.model_dir, "test_predictions.npz")
    images, labels = load_predictions(pred_file)

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
    metadata_file = open(os.path.join(FLAGS.log_dir, 'metadata.tsv'), 'w')
    if FLAGS.znist:
        create_znist_metadata(labels, metadata_file)
    else:
        create_mnist_metadata(labels, metadata_file)
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

    print("Run: tensorboard --logdir {}".format(FLAGS.log_dir))
    print("Open: http://localhost:6006")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to input data directory')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to log directory')
    parser.add_argument('--create_sprite', action='store_true',
                        help='Create sprite image')
    parser.add_argument('--znist', action='store_true',
                        help='Create znist metadata file')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
