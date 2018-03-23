from utilis.cost_functions import *
import numpy as np
import time


class FCNeuralNet:
    def __init__(self, layers, cost_func=subtract_err):
        self.layers = layers
        self.cost_func = cost_func

    def print_network(self):
        print("network:")
        for layer in self.layers:
            print("layer - %s: weights: %s" % (layer.name, layer.weights.shape))

    def train(self, x, y, learning_rate=0.01, epochs=10000,
              batch_size=256, print_epochs=1000,
              monitor_train_accuracy=False):
        print("training start")
        start_time = time.time()

        for k in range(epochs):
            rand_indices = np.random.randint(x.shape[0], size=batch_size)
            batch_x = x[rand_indices]
            batch_y = y[rand_indices]

            results = self.feed_forward(batch_x)

            error = self.cost_func(results[-1], batch_y)

            if (k+1) % print_epochs == 0:
                loss = np.mean(np.abs(error))
                msg = "epochs: {0}, loss: {1:.4f}".format((k+1), loss)
                if monitor_train_accuracy:
                    accuracy = self.accuracy(x, y)
                    msg += ", accuracy: {0:.2f}%".format(accuracy)
                print(msg)
            deltas = self.back_propagate(results, error)
            self.update_weights(results, deltas, learning_rate)

        end_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
        print("training complete, elapsed time:", elapsed_time)

    def feed_forward(self, x):
        results = [x]
        for i in range(len(self.layers)):
            output_result = self.layers[i].get_output(results[i])
            results.append(output_result)
        return results

    def back_propagate(self, results, error):
        last_layer = self.layers[-1]
        deltas = [error * last_layer.activation(results[-1], derivative=True)]

        # we need to begin at the second to last layer - (a layer before the output layer)
        for i in range(len(results) - 2, 0, -1):
            layer = self.layers[i]
            delta = deltas[-1].dot(layer.weights.T) * layer.activation(results[i], derivative=True)
            deltas.append(delta)

        deltas.reverse()
        return deltas

    def update_weights(self, results, deltas, learning_rate):
        # 1. Multiply its output delta and input activation to get the gradient of the weight.
        # 2. Subtract a ratio (percentage) of the gradient from the weight.
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_result = results[i]
            delta = deltas[i]
            layer.weights -= learning_rate * layer_result.T.dot(delta)
            # layer.biases += delta

    def predict(self, x):
        return self.feed_forward(x)[-1]

    def accuracy(self, x_data, y_labels):
        predictions = np.argmax(self.predict(x_data), axis=1)
        labels = np.argmax(y_labels, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy * 100
