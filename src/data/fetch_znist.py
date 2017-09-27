"""
Download ZNIST data
"""

import urllib
import os

directory = "data/znist"
url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

datasets = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

if not os.path.exists(directory):
    os.makedirs(directory)

for set_name in datasets:
    urllib.urlretrieve(os.path.join(url, set_name),
                       os.path.join(directory, set_name))
