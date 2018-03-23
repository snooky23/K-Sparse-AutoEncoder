import numpy as np
import os
import urllib
import gzip
import struct


class mnist_helper:

    def __init__(self):
        self.train_lbl, self.train_img, self.test_lbl, self.test_img = self.load_mnist_data()

    def get_data(self):
        return self.train_lbl, self.train_img, self.test_lbl, self.test_img

    @staticmethod
    def download_data(url, force_download=False):
        fname = url.split("/")[-1]
        if force_download or not os.path.exists(fname):
            urllib.request.urlretrieve(url, fname)
        return fname

    def load_data(self, label_url, image_url, force_download=False):
        with gzip.open(self.download_data(label_url, force_download)) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.int8)
        with gzip.open(self.download_data(image_url, force_download), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
        return label, image

    def load_mnist_data(self):
        path = 'http://yann.lecun.com/exdb/mnist/'
        train_lbl, train_img = self.load_data(
            path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
        test_lbl, test_img = self.load_data(
            path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')

        return self.to_one_hot(train_lbl), train_img, self.to_one_hot(test_lbl), test_img

    @staticmethod
    def to_one_hot(labels, num_classes=10):
        return np.eye(num_classes)[labels]
