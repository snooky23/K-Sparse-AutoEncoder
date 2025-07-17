"""Main script for MNIST classification and K-sparse autoencoder experiments.

This script provides functionality to train and evaluate both:
1. MNIST digit classification using a standard neural network
2. K-sparse autoencoder for MNIST digit reconstruction
"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from layers.linear_layer import LinearLayer
from layers.sparse_layer import SparseLayer
from nets.fcnn import FCNeuralNet
from utilis.activations import sigmoid_function
from utilis.cost_functions import subtract_err
import utilis.mnist.mnist_helper as mh


def run_mnist_predictions() -> None:
    """Train and evaluate MNIST digit classification."""
    helper = mh.MnistHelper()
    y_labels, train_img, test_lbl, test_img = helper.get_data()
    # print_samples(train_img, train_lbl, n_samples=10)

    img_size = 28
    learning_rate = 0.01
    epochs = 10000
    batch_size = 256
    cost_func = subtract_err

    x_data = train_img.reshape(-1, img_size * img_size) / np.float32(256)

    layers = [
        LinearLayer(name="input", n_in=x_data.shape[1], n_out=30, activation=sigmoid_function),
        LinearLayer(name="hidden 1", n_in=30, n_out=10, activation=sigmoid_function),
        LinearLayer(name="output", n_in=10, n_out=y_labels.shape[1], activation=sigmoid_function)
    ]

    nn = FCNeuralNet(layers=layers, cost_func=cost_func)
    nn.print_network()

    nn.train(x_data, y_labels, learning_rate=learning_rate, epochs=epochs,
             batch_size=batch_size, print_epochs=1000,
             monitor_train_accuracy=True)

    test_data = test_img.reshape(-1, img_size * img_size) / np.float32(256)
    accuracy = nn.accuracy(test_data, test_lbl)
    print("test accuracy: {0:.2f}%".format(accuracy))


def run_auto_encoder() -> None:
    """Train and evaluate K-sparse autoencoder on MNIST."""
    img_size = 28
    num_hidden = 100
    k = 10
    learning_rate = 0.01
    epochs = 3000
    batch_size = 256
    print_epochs = 1000
    num_test_examples = 10

    helper = mh.MnistHelper()
    train_lbl, train_img, test_lbl, test_img = helper.get_data()

    x_data = train_img.reshape(-1, img_size * img_size) / np.float32(256)
    test_data = test_img.reshape(-1, img_size * img_size) / np.float32(256)

    layers = [
        # LinearLayer(name="input", n_in=x_data.shape[1], n_out=num_hidden, activation=sigmoid_function),
        SparseLayer(name="hidden 1", n_in=x_data.shape[1], n_out=num_hidden,
                    activation=sigmoid_function, num_k_sparse=k),
        LinearLayer(name="output", n_in=num_hidden, n_out=x_data.shape[1], activation=sigmoid_function)
    ]

    nn = FCNeuralNet(layers=layers, cost_func=subtract_err)
    nn.print_network()

    nn.train(x_data, x_data, learning_rate=learning_rate, epochs=epochs,
             batch_size=batch_size, print_epochs=print_epochs)

    # Encode and decode images from test set and visualize their reconstruction.
    n = num_test_examples

    test_samples = test_data[0:n]
    encode_samples = nn.layers[0].weights.T
    output_samples = nn.predict(test_samples)

    print("encode_samples", encode_samples.shape)
    print("Output shape", output_samples.shape)

    img_input = test_samples.reshape(-1, img_size, img_size)
    img_encode = encode_samples.reshape(-1, img_size, img_size)
    img_output = output_samples.reshape(-1, img_size, img_size)

    title = "k-sparse auto encoder for k={0}, epochs={1}, batch_size={2}".format(k, epochs, batch_size)
    all_images = np.concatenate((img_input, img_output, img_encode))

    add_plot_images(all_images, cols=10, img_size=img_size, title=title)

    plt.show()


def add_plot_images(images: np.ndarray, cols: int = 10, img_size: int = 28, title: str = None) -> None:
    """Plot a grid of images.
    
    Args:
        images: Array of images to plot
        cols: Number of columns in the grid
        img_size: Size of each image
        title: Optional title for the plot
    """
    n_images = len(images)
    fig = plt.figure(figsize=(img_size, img_size))
    plt.title(title, fontsize=24)
    # plt.gray()
    rows = int(n_images / cols)
    for i in range(1, n_images + 1):
        img = images[i - 1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)


def main() -> None:
    """Main function to choose between classification and autoencoder tasks."""
    is_auto_encoder = False
    # is_auto_encoder = True
    if is_auto_encoder:
        run_auto_encoder()
    else:
        run_mnist_predictions()


if __name__ == '__main__':
    main()
