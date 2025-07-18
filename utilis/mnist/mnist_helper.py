"""MNIST dataset loader and helper utilities.

This module provides functionality to download and load the MNIST dataset
for handwritten digit recognition tasks.
"""
from typing import Tuple
import numpy as np
import os
import urllib.request
import gzip
import struct


class MnistHelper:
    """Helper class for loading and processing MNIST dataset."""

    def __init__(self) -> None:
        """Initialize MNIST helper and load dataset."""
        self.train_lbl, self.train_img, self.test_lbl, self.test_img = self.load_mnist_data()

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the loaded MNIST dataset.
        
        Returns:
            Tuple of (train_labels, train_images, test_labels, test_images)
        """
        return self.train_lbl, self.train_img, self.test_lbl, self.test_img

    @staticmethod
    def download_data(url: str, force_download: bool = False) -> str:
        """Download data file from URL.
        
        Args:
            url: URL to download from
            force_download: Force re-download even if file exists
            
        Returns:
            Local filename of downloaded file
        """
        fname = url.split("/")[-1]
        if force_download or not os.path.exists(fname):
            urllib.request.urlretrieve(url, fname)
        return fname

    def load_data(self, label_url: str, image_url: str, force_download: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Load labels and images from MNIST data files.
        
        Args:
            label_url: URL for label file
            image_url: URL for image file
            force_download: Force re-download of files
            
        Returns:
            Tuple of (labels, images)
        """
        with gzip.open(self.download_data(label_url, force_download)) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.frombuffer(flbl.read(), dtype=np.int8)
        with gzip.open(self.download_data(image_url, force_download), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
        return label, image

    def load_mnist_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load complete MNIST dataset.
        
        Returns:
            Tuple of (train_labels_onehot, train_images, test_labels_onehot, test_images)
        """
        # Use working mirror URLs for MNIST dataset
        path = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        train_lbl, train_img = self.load_data(
            path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
        test_lbl, test_img = self.load_data(
            path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')

        return self.to_one_hot(train_lbl), train_img, self.to_one_hot(test_lbl), test_img

    @staticmethod
    def to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """Convert integer labels to one-hot encoding.
        
        Args:
            labels: Integer labels array
            num_classes: Number of classes (default 10 for MNIST)
            
        Returns:
            One-hot encoded labels
        """
        return np.eye(num_classes)[labels]
