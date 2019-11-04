import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split


class CrossEntropy:
    @staticmethod
    def function(a, y):
        return np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a))

    @staticmethod
    def derivative(a, y):
        return a - y


class MSE:
    @staticmethod
    def function(a, y):
        return np.mean((a - y) ** 2)

    @staticmethod
    def derivative(a, y):
        return a - y


class NeuralNetwork:
    """
    Info

    Parameters
    ----------
    layers : list of integers
        List of layers where layers[i] is the number of nodes in layers i.
        Note: layers should include:
            * input layer
            * hidden layers
            * output layer

    Attributes
    ----------
    num_layers : int
        Number of layers. Equivalent to len(layers).

    weights : list of arrays
        Length of list is (num_layers - 1) and weights[i] contains and array
        of shape(layers[i+1], layers[i]).

    biases : list of arrays
        Length of list is (num_layers - 1) and biases[i] contains and array
        of shape(layers[i], ).

    Example
    -------
    NN = NeuralNetwork([20, 15, 10, 1])
    NN.SGD(X, y, epoch=10, batch_size=10, learning_rate=0.1)
    y_pred = NN.predict(X_test)
    """

    def __init__(self):
        pass

    def sigmoid(self, z):
        """Sigmoid function as activation function"""
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(selv, sigmoid):
        return sigmoid * (1 - sigmoid)

    def SGD(self, X, y, layers, cost, epochs=50, batch_size=100, eta=0.01, reg=1e-6):
        """
        Info

        Parameters
        ----------
        X : array, shape(N_samples, N_features)
            Matrix containing all input data.

        y : array, shape(N_samples, )
            y data with training data output values.

        epochs : int
            Number of total iterations (over all data points). This is the
            outer loop in the algorithm.

        batch_size : int
            Size of each mini batch. The algorithm does one mini batch at
            a time and averages over those results. Larger batch
        """
        X, X_vali, y, y_vali = train_test_split(X, y, test_size=0.10)
        n, p = X.shape
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = np.array(
            [np.random.normal(0, 0.5, (i, j)) for i, j in zip(layers[:-1], layers[1:])]
        )
        self.biases = np.array([np.zeros(j) for j in layers[1:]])
        self.vali_accuracy = np.zeros(epochs)
        num_batches = int(n / batch_size)
        batches = np.random.permutation(num_batches * batch_size)
        batches = batches.reshape(num_batches, batch_size)
        print(f"\tTraining Neural Network with {epochs} epochs")
        self.printprogress(0)
        # Loop over epochs
        for e in range(epochs):
            # Epoch >>>
            batches = np.random.permutation(n)
            nabla_b = np.array(
                [np.zeros(b.shape) for b in self.biases], dtype=np.ndarray
            )
            nabla_w = np.array(
                [np.zeros(w.shape) for w in self.weights], dtype=np.ndarray
            )
            # loop over all mini batches
            for b in range(num_batches):
                batch = batches[batch_size * b : batch_size * (b + 1)]
                Xi = X[batch]
                yi = y[batch, np.newaxis]
                # Feedforward >>>
                a = Xi
                A = [a]
                for w, b in zip(self.weights, self.biases):
                    a = self.sigmoid(a @ w + b)
                    A.append(a)
                # <<< Feedforward
                # Backpropagation >>>
                # Last layer, delta_L:
                delta_l = cost.derivative(A[-1], yi) * self.sigmoid_derivative(A[-1])
                nabla_b[-1] = np.mean(delta_l, axis=0)
                nabla_w[-1] = A[-2].T @ delta_l
                # Second to last down to first layer
                for l in range(2, self.num_layers):
                    delta_l = (
                        delta_l @ self.weights[-l + 1].T
                    ) * self.sigmoid_derivative(A[-l])
                    nabla_b[-l] += np.mean(delta_l, axis=0)
                    nabla_w[-l] += A[-l - 1].T @ delta_l
                    # self.loss[e] += cost.function(A[-1], yi)
                # Update weights and biases (L2 regularization)
                self.weights = (
                    self.weights * (1 - eta * reg / n) - eta * nabla_w / batch_size
                )
                self.biases = self.biases - eta * nabla_b / batch_size
            # <<< Backpropagation
            y_pred = self.predict(X_vali)
            self.vali_accuracy[e] = np.sum(y_pred == y_vali) / y_vali.shape[0]
            self.printprogress((e + 1) / epochs)
            # <<< Epoch

        # weights and biases are now trained
        return None

    def predict(self, X, binary=True):
        y_pred = np.zeros(X.shape[0])
        a = X
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(a @ w + b)
        y_pred = a.reshape(-1)
        if binary:
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
        return y_pred

    def load_weights_and_biases(self, layers):
        """
        Load weights and biases saved in data/NN...name.../ + b/ and w/.
        These folders contain .npz files with biases and weights respectively.

        Parameter
        ---------
        layers : list of int
            List of nodes per layers, including input and output layer.
        """
        path = "./data/NN"
        self.layers = layers
        for l in layers:
            path += "_" + str(l)
        if not os.path.exists(path):
            raise ValueError("Path for these layers does not exist!")
        wpath = path + "/w/"
        bpath = path + "/b/"
        self.weights = np.array(
            [np.load(wpath + str(i) + ".npz")["w"] for i in range(len(layers) - 1)]
        )
        self.biases = np.array(
            [np.load(bpath + str(i) + ".npz")["b"] for i in range(len(layers) - 1)]
        )
        print("\tWeights and biases loaded from " + path + "/")

    def save_weights_and_biases(self):
        """
        Save weights and biases after training (typically after SGD has run).
        The files are saved in data/NN...name.../ + b/ and w/.
        """
        path = "./data/NN"
        for l in self.layers:
            path += "_" + str(l)
        wpath = path + "/w"
        bpath = path + "/b"
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(wpath)
            os.mkdir(bpath)
        for i, w, b in zip(range(len(self.layers) - 1), self.weights, self.biases):
            np.savez(os.path.join(wpath, str(i) + ".npz"), w=w)
            np.savez(os.path.join(bpath, str(i) + ".npz"), b=b)
        print("\tWeights and biases saved in " + path + "/")

    def printprogress(self, progress):
        """
        Prints a progress bar during the iterative algorithm.
        """
        sys.stdout.flush()
        width = 30
        status = ""
        if progress >= 1:
            progress = 1
            status = "Done.\r\n"
        bl = "#"
        li = "-"
        nbl = int(round(width * progress))
        text = f"\r\t[{nbl*bl + (width-nbl)*li}] {progress*100:3.2f} % {status}"
        sys.stdout.write(text)
        sys.stdout.flush()
