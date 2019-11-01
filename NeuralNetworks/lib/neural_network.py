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

    def d_cost(self, f, y):
        """Same for MSE and Xentropy"""
        return f - y

    def sigmoid(self, z):
        """Sigmoid function as activation function"""
        return 1 / (1 + np.exp(-z))

    def SGD(
        self,
        X,
        y,
        layers,
        epochs=50,
        batch_size=100,
        eta=0.01,
        cost_function="xentropy",
        reg=1e-6
    ):
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
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = np.array(
            [np.random.normal(0, 0.5, (i, j)) for i, j in zip(layers[:-1], layers[1:])]
        )
        self.biases = np.array([np.zeros(j) for j in layers[1:]])
        num_batches = int(n / batch_size)
        batches = np.random.permutation(num_batches * batch_size)
        batches = batches.reshape(num_batches, batch_size)
        nabla_b = np.array([np.zeros(b.shape) for b in self.biases], dtype=np.ndarray)
        nabla_w = np.array([np.zeros(w.shape) for w in self.weights], dtype=np.ndarray)
        print(f"\tTraining Neural Network with {epochs} epochs")
        self.print_progress(0)

        for e in range(epochs):
            # Epoch >>>
            batches = np.random.permutation(n)
            # loop over all mini batches
            for b in range(num_batches):
                batch = batches[batch_size * b : batch_size * (b + 1)]
                Xi = X[batch]
                yi = y[batch,np.newaxis]
                # Feedforward >>>
                a = Xi
                A = [a]
                for w, b in zip(self.weights, self.biases):
                    a = self.sigmoid(a @ w + b)
                    A.append(a)
                # <<< Feedforward
                # Backpropagation >>>
                # Last layer, delta_L:
                delta_l = self.d_cost(A[-1], yi) * (A[-1] * (1 - A[-1]))
                nabla_b[-1] = np.mean(delta_l, axis=0)
                nabla_w[-1] = A[-2].T @ delta_l
                # Second to last down to first layer
                for l in range(2, self.num_layers):
                    delta_l = (delta_l @ self.weights[-l + 1].T) * (A[-l] * (1 - A[-l]))
                    nabla_b[-l] += eta*np.mean(delta_l, axis=0)
                    nabla_w[-l] += eta*A[-l - 1].T @ delta_l
                # Update weights and biases
                self.weights -= nabla_w / batch_size - reg*self.weights
                self.biases -= nabla_b / batch_size
                # <<< Backpropagation

            self.print_progress((e + 1) / epochs)
            # <<< Epoch

        # weights and biases are now trained
        return None

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        a = X
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(a @ w + b)

        y_pred[a.reshape(-1,) < 0.5] = 0
        y_pred[a.reshape(-1,) >= 0.5] = 1
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

    def print_progress(self, progress):
        """
        Prints a progress bar and percentage towards completion during
        training in SGD
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
