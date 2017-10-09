"""
Reduce dimensionality of feature space extracted from trained ConvNet model
with PCA and t-SNE
"""

from __future__ import print_function
from sklearn.manifold import TSNE
import numpy as np
import tf_pca
import utils
import argparse
import os

FLAGS = None
LEARNING_RATE = 200
PERPLEXITY = 40
ANGLE = 0.2


def load_features(file):
    d = np.load(file)
    images = d['images']
    feats = d['features']
    labels = d['labels']
    return images, feats, labels


def performPCA(dim, data, labels):
    '''
    Return data with PCA reduced dimensionality
    '''
    # Note: sklearn's PCA implementation throws seg fault when reducing large
    # dataset to dimensions > 20 on mac, to avoid this PCA is implemented
    # directly using tensorflow
    # PCA:
    # - linear and fast dimensionality reduction method
    # - reduces number of dimensions in a dataset while retaining most
    # important X% of data
    # - same data will always produce same result
    pca = tf_pca.PCA(data, labels)
    pca.fit()
    return pca.reduce(n_dimensions=dim)


def performTSNE(dim, data, labels):
    '''
    Return data with t-SNE reduced dimensionality
    '''
    # t-SNE:
    # - nonlinearly embeds points into lower dimensional space by preserving
    # structure of neighbours as much as possible
    # - scales quadratically ~ O(n^2) hence memory/time limitations
    # - learning_rate defines step size for iterations
    # - perplexity balances the focus between local and global structure
    # of data in the optimization process
    # - higher perplexity means a data point will consider more points as its
    # close neighbors and lower means less (suggested: 5 > perplexity > 50)
    # - angle controls the speed vs accuracy tradeoff
    # - lower angle means higher accuracy but slower computation
    tsne = TSNE(n_components=dim, learning_rate=LEARNING_RATE,
                perplexity=PERPLEXITY, angle=ANGLE, verbose=2)
    return tsne.fit_transform(data)


def embed_features(feats, labels):
    '''
    Embed data using PCA and t-SNE so that points near each other in original
    data will be near each other in embedding
    '''
    # The dim of feature space is too high to use t-SNE directly (65000x1024)
    # so we must first reduce dim with PCA to 65000x50 before we can perform
    # t-SNE
    reduced_feats = performPCA(50, feats, labels)
    embedded_feats = performTSNE(2, reduced_feats, labels)
    return embedded_feats


def save_embedding(file_name, images, labels, embedded_feats):
    file = os.path.join(FLAGS.model_dir, file_name)
    np.savez_compressed(file, images=images, labels=labels,
                        embedded_features=embedded_feats)


def main():
    train_data = os.path.join(FLAGS.model_dir, "train_features.npz")
    test_data = os.path.join(FLAGS.model_dir, "test_features.npz")

    # Load features
    train_images, train_feats, train_labels = load_features(train_data)
    test_images, test_feats, test_labels = load_features(test_data)

    # Combine data to perform dimensionality reduction on entire data set as
    # sklearn's t-sne implementation does not support batch/streaming
    feats = np.concatenate((train_feats, test_feats))
    labels = np.concatenate((train_labels, test_labels))

    print("Reducing Dimensionality of Feature Space:")
    embedded_feats = embed_features(feats, labels)

    # Split training and test data again and save to file
    split = train_feats.shape[0]
    embedded_train_feats = embedded_feats[:split]
    embedded_test_feats = embedded_feats[split:]

    # # Save embeddings to .npz file
    save_embedding("embedded_train_feats.npz", train_images, train_labels,
                   embedded_train_feats)
    save_embedding("embedded_test_feats.npz", test_images, test_labels,
                   embedded_test_feats)
    print("Embedded features extracted to dir: {}".format(FLAGS.model_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default=utils.git_hash_dir_path('models/conv_mnist'),
                        help='Path to features directory')

    FLAGS, unparsed = parser.parse_known_args()
    main()
