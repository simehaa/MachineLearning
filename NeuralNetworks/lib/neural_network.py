import numpy as np
import sys
import os


class CrossEntropy:
    """May be used as cost function in the NeuralNetwork class. Both the function
    itself and the derivative are provided as methods."""
    @staticmethod
    def function(a, y):
        return np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a))

    @staticmethod
    def derivative(a, y):
        return a - y


class MSE:
    """May be used as cost function in the NeuralNetwork class. Both the function
    itself and the derivative are provided as methods."""
    @staticmethod
    def function(a, y):
        return np.mean((a - y) ** 2)

    @staticmethod
    def derivative(a, y):
        return a - y


def sigmoid(z):
    """Sigmoid function as activation function"""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(s):
    """Sigmoid derivative, takes sigmoid as input"""
    return s * (1 - s)


def linear(z):
    """Linear activation function"""
    return z


def linear_derivative(z):
    """Linear activation function derivative"""
    return 1


def tanh(z):
    """tanh as activation function"""
    return np.tanh(z)


def tanh_derivative(t):
    """tanh derivative, takes tanh as input"""
    return 1 - t * t


def relu(z):
    """ReLU activation function"""
    a = np.zeros(z.shape)
    a[z >= 0] = z[z >= 0]
    return a


def relu_derivative(z):
    """ReLU derivative"""
    a = np.zeros(z.shape)
    a[z >= 0] = 1
    return a


def l_relu(z):
    """ReLU activation function"""
    a = 0.01*z
    a[z >= 0] = z[z >= 0]
    return a


def l_relu_derivative(z):
    """ReLU derivative"""
    a = np.ones(z.shape)
    a[z < 0] = 0.01
    return a


def relu6(z):
    """ReLU6 activation function"""
    a = np.zeros(z.shape)
    a[z >= 0] = z[z >= 0]
    a[z >= 6] = 6
    return a


def relu6_derivative(z):
    """ReLU6 derivative"""
    a = np.zeros(z.shape)
    a[np.logical_or(z >= 0, z <= 6)] = 1
    return a


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

    def __init__(self, cost, layers, act_fns):
        self.layers = layers
        self.num_layers = len(layers)
        self.act = []
        self.d_act = []
        self.cost = cost
        if len(layers) - 1 != len(act_fns):
            raise ValueError("Wrong number of activation functions")
        for a in act_fns:
            if a == "sigmoid":
                self.act.append(sigmoid)
                self.d_act.append(sigmoid_derivative)
            elif a == "linear":
                self.act.append(linear)
                self.d_act.append(linear_derivative)
            elif a == "tanh":
                self.act.append(tanh)
                self.d_act.append(tanh_derivative)
            elif a == "relu":
                self.act.append(relu)
                self.d_act.append(relu_derivative)
            elif a == "relu6":
                self.act.append(relu6)
                self.d_act.append(relu6_derivative)
            elif a == "l_relu":
                self.act.append(l_relu)
                self.d_act.append(l_relu_derivative)
            else:
                raise ValueError(a + " is not implemented as an activation function.")

    def SGD(self, X, y, validation_data, epochs=50, batch_size=100, eta=0.01, reg=1e-6, save_frames=False):
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
        self.cost_arr = np.zeros(epochs)
        self.weights = np.array(
            [
                np.random.normal(0, 0.5, (i, j))
                for i, j in zip(self.layers[:-1], self.layers[1:])
            ]
        )
        self.X_test = validation_data[0]
        self.y_test = validation_data[1]
        self.biases = np.array([np.zeros(j) for j in self.layers[1:]])
        num_batches = int(n / batch_size)
        batches = np.random.permutation(num_batches * batch_size)
        batches = batches.reshape(num_batches, batch_size)
        print(f"\tTraining Neural Network with {epochs} epochs")
        self.printprogress(0, 0)
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
                for w, b, fn in zip(self.weights, self.biases, self.act):
                    a = fn(a @ w + b)
                    A.append(a)
                # <<< Feedforward
                # Backpropagation >>>
                # Last layer, delta_L:
                delta_l = self.cost.derivative(A[-1], yi) * self.d_act[-1](A[-1])
                nabla_b[-1] = np.mean(delta_l, axis=0)
                nabla_w[-1] = A[-2].T @ delta_l
                # Second to last down to first layer
                for l in range(2, self.num_layers):
                    delta_l = (delta_l @ self.weights[-l + 1].T) * self.d_act[-l](A[-l])
                    nabla_b[-l] += np.mean(delta_l, axis=0)
                    nabla_w[-l] += A[-l - 1].T @ delta_l
                    # self.loss[e] += cost.function(A[-1], yi)
                # Update weights and biases (L2 regularization)
                self.weights = (
                    self.weights * (1 - eta * reg / n) - eta * nabla_w / batch_size
                )
                self.biases = self.biases - eta * nabla_b / batch_size
            # <<< Backpropagation
            self.printprogress((e + 1) / epochs, e)

            if save_frames:# Used to save the predictions for animation purposes
                np.savez("./data/frames/frame" + str(e) + ".npz", y=self.predict(self.X_test))
                
            # <<< Epoch
        print("\n")
        # weights and biases are now trained
        return None

    def predict(self, X, binary=False):
        """
        Evaluate output y, given X. Essentially the feed forward algorithm.

        Parameters
        ----------
        X : array, shape(N, p)
            Input layer with N data points and p features.

        binary : bool
            If True, the output y will convert all elements above 0.5 to 1 and
            all elements below 0.1 to 0, resulting in binary outputs.

        Returns
        -------
        y_pred : array, shape(N, )
            The output vector.
        """
        y_pred = np.zeros(X.shape[0])
        a = X
        for w, b, fn in zip(self.weights, self.biases, self.act):
            a = fn(a @ w + b)
        y_pred = a.reshape(-1)
        if binary:
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
        return y_pred

    def printprogress(self, progress, epoch):
        """
        Prints a progress bar during the iterative algorithm.
        """
        sys.stdout.flush()
        width = 30
        y = self.y_test
        cost = self.cost.function(self.predict(self.X_test), self.y_test)
        self.cost_arr[epoch] = cost
        bl = "#"
        li = "-"
        nbl = int(round(width * progress))
        text = (
            f"\r\t[{nbl*bl + (width-nbl)*li}] {progress*100:3.2f} %" + f" Cost = {cost:2.4f}"
        )
        sys.stdout.write(text)
        sys.stdout.flush()
