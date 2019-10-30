import numpy as np
import sys
import os


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

    def load_weights_and_biases(self, layers):
        path = "./data/NN"
        for l in layers:
            path += "_" + str(l)
        if not os.path.exists(path):
            raise ValueError("Path for these layers does not exist!")
        wpath = path + "/w/"
        bpath = path + "/b/"
        self.weights = []
        self.biases = []
        for l in layers[:-1]:
            w = np.load(wpath + str(l) + ".npz")["w"]
            b = np.load(bpath + str(l) + ".npz")["b"]
            self.weights.append(w)
            self.biases.append(b)
        print("\tWeights and biases loaded from " + path + "/")


    def save_weights_and_biases(self):
        path = "./data/NN"
        for l in self.layers:
            path += "_" + str(l)
        wpath = path + "/w"
        bpath = path + "/b"
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(wpath)
            os.mkdir(bpath)
        for l, w, b in zip(self.layers[:-1], self.weights, self.biases):
            np.savez(os.path.join(wpath, str(l) + ".npz"), w=w)
            np.savez(os.path.join(bpath, str(l) + ".npz"), b=b)
        print("\tWeights and biases saved in " + path + "/")

    def set_layers(self, layers):
        """
        Parameters
        ----------
        layers : list
            List containing number of nodes per hidden layer AND nodes in
            output layer.
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = [
            np.random.normal(0, 0.5, (j, i))
            for i, j in zip(layers[:-1], layers[1:])
        ]
        self.biases = [np.random.normal(0, 0.5, (j, 1)) for j in layers[1:]]

    def SGD(self, X, y, epochs=50, batch_size=100, learning_rate=0.01):
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

        n, p = X.shape

        num_batches = int(n / batch_size)
        batches = np.random.permutation(num_batches * batch_size)
        batches = batches.reshape(num_batches, batch_size)

        print(f"\tTraining Neural Network with {epochs} epochs")
        self.print_progress(0)

        for e in range(epochs):
            batches = np.random.permutation(n)
            # loop over all mini batches
            for b in range(num_batches):
                # Inlucde all data ONCE approach:
                batch = batches[batch_size * b : batch_size * (b + 1)]
                # Bootstrap approach:
                # batch = np.random.permutation(n)[:batch_size]
                Xi = X[batch]
                yi = y[batch]
                Z = [Xi]
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                # loop over data points in mini batch
                for _x, _y in zip(Xi, yi):
                    delta_nabla_w, delta_nabla_b = self.back_propagate(_x, _y)
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

                self.weights = [
                    w - nw * learning_rate / batch_size
                    for w, nw in zip(self.weights, nabla_w)
                ]
                self.biases = [
                    b - nb * learning_rate / batch_size
                    for b, nb in zip(self.biases, nabla_b)
                ]

            self.print_progress((e + 1) / epochs)

        # weights and biases are now trained
        return None

    def back_propagate(self, X, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        f = X[:, np.newaxis]
        fs = [f]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = w @ f + b
            f = self.sigmoid(z)
            zs.append(z)
            fs.append(f)

        diff = self.cost_derivative(fs[-1], y)
        nabla_w[-1] = diff @ fs[-1].T
        nabla_b[-1] = diff

        for l in range(2, self.num_layers):
            nabla_activation = self.sigmoid_derivative(zs[-l])
            diff = nabla_activation * (self.weights[-l + 1].T @ diff)
            nabla_w[-l] = (fs[-l - 1] @ diff.T).T
            nabla_b[-l] = diff

        return nabla_w, nabla_b

    def cost_derivative(self, f, y):
        return (f - y)

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s*(1 - s)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i, xi in enumerate(X):
            f = xi[:, np.newaxis]
            for w, b in zip(self.weights, self.biases):
                z = w @ f + b
                f = self.sigmoid(z)
            y_pred[i] = float(f)

        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1

        return y_pred

    def print_progress(self, progress):
        """Prints a progress bar and percentage towards completion during
        training in SGD"""
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
