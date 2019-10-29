class NeuralNetwork:
    """
    Info

    Parameters
    ----------
    b :

    Attributes
    ----------
    b :
    """

    def __init__(self, layers):
        """
        Info

        Parameters
        ----------
        layers : list
            List containing number of nodes per hidden layer AND nodes in
            output layer.

        Attributes
        ----------

        """
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = [
            np.random.normal(0, 0.5, (j, i)) for i, j in zip(layers[:-1], layers[1:])
        ]
        self.biases = [np.random.normal(0, 0.5, (i, 1)) for j in layers[:-1]]

    def train(self, epochs=50, batch_size=100, learning_rate=0.01):
        """
        Info

        Parameters
        ----------
        X : array, shape(N_samples, N_features)
            Matrix containing all input data.

        y : array, shape(N_samples, )
            y data with training data output values.
        """

        Xn, Xp = X.shape
        yn, yp = y.shape

        num_batches = int(Xn / batch_size)
        batches = np.random.permutation(num_batches * batch_size)
        batches = batches.reshape(num_batches, batch_size)

        for e in range(epochs):
            batches = np.random.permutation(Xn)
            for b in range(num_batches):
                batch = batches[batch_size * b : batch_size * (b + 1)]
                Xi = X[batch]
                yi = y[batch]
                Z = [Xi]

        return weights, biases

    def feedforward(self, Xi):
        """Feed forward from input to output with activation function tanh"""
        for W, B in zip(self.weights, self.biases):
            Xi = np.tanh(W @ Xi + B)
        return Xi
