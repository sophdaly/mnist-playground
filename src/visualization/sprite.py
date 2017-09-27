"""
Helper functions for creating sprite images
"""

import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_SIZE = 28


def create_sprite(input_images, log_dir, invert=True):
    '''
    Creates sprite image file
    '''
    images = _reshape_digits(input_images)
    if invert:
        images = _invert_greyscale(images)
    sprite = _sprite_image(images)

    # Display and save sprite
    plt.imsave(os.path.join(log_dir, 'sprite.png'), sprite, cmap='gray')


def _sprite_image(images):
    '''
    Returns sprite image of input images
    '''
    if isinstance(images, list):
        images = np.array(images)
    height = images.shape[1]
    width = images.shape[2]
    no_plots = int(np.ceil(np.sqrt(images.shape[0])))

    sprite = np.ones((height * no_plots, width * no_plots))

    for i in range(no_plots):
        for j in range(no_plots):
            filter = i * no_plots + j
            if filter < images.shape[0]:
                image = images[filter]
                sprite[i * height:(i + 1) * height,
                       j * width:(j + 1) * width] = image

    return sprite


def _reshape_digits(digits):
    '''
    Reshapes mnist digit [batch, 28 * 28] to matrix [batch, 28, 28]
    '''
    return np.reshape(digits, (-1, IMAGE_SIZE, IMAGE_SIZE))


def _invert_greyscale(digits):
    return 1 - digits
