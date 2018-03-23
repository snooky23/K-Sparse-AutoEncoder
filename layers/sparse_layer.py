from layers.linear_layer import LinearLayer
from utilis.activations import *


class SparseLayer(LinearLayer):

    def __init__(self, name, n_in, n_out, activation=sigmoid_function, num_k_sparse=10):
        LinearLayer.__init__(self, name, n_in, n_out, activation)
        self.num_k_sparse = num_k_sparse

    def get_output(self, x):
        result = self.activation(x.dot(self.weights) + self.biases)

        k = self.num_k_sparse
        if k < result.shape[1]:
            for raw in result:
                indices = np.argpartition(raw, -k)[-k:]
                mask = np.ones(raw.shape, dtype=bool)
                mask[indices] = False
                raw[mask] = 0

        self.result = result
        return result
