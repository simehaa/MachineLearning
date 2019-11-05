import numpy as np
import sys


class SGDClassification:
    """
    Classification by Stochastic Gradient Descent with mini batches.
    Iterative method where the parameter vector beta is updated by

        beta_new = beta - gamma X^T (y - p).

    gamma is the learning rate and it is 1/10, 1/11, 1/12, ... (decreasing).
    For each epoch, this is done for a random mini batch, num_batches times.

    Parameters
    ----------
    batch_size : int
        Size of each mini batch.

    epochs : int
        Number of total iterations.

    Attributes
    ----------
    features : int
        Number of features used in fit() method. It if found by the number
        of columns in X.

    beta : array, shape(features, )
        Parameter vector which is found after fit() method has been called.

    Example
    -------
    clf = SGDClassification(100, 1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    """

    def __init__(self, batch_size=100, epochs=1000):
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y, beta=None):
        """
        Perform SGD with mini batches on X, y as training data. if initial
        guess, beta is not provided, then this will be a random guess with
        values in [- 0.7, + 0.7].

        Parameters
        ----------
        X : array, shape (N, features)
            Training data feature matrix.

        y : array, shape (N, )
            Training data output values.

        beta : array, shape(features, )
            Initial guess for parameter vector, size num_features.

        Returns
        -------
        None
            features and beta are now new attributes and predict()
            can be called.
        """
        N, self.features = X.shape
        if beta == None:
            self.beta = (np.random.random(self.features) - 0.5) * 1.4
        else:
            if beta.shape[0] == self.features:
                self.beta = beta
            else:
                raise ValueError(
                    "beta must be of the same shape as number of columns in X"
                )
        batch_size = self.batch_size
        num_batches = int(N / batch_size)
        learning_rate = lambda t: 1 / (t + 10)
        self.bias = 0

        self.printprogress(0)
        # iterative loop over epochs
        for t in range(self.epochs):
            # loop over mini batches
            batches = np.random.permutation(N)
            for b in range(num_batches):
                # pick a random batch, num_batches times
                batch = batches[batch_size * b : batch_size * (b + 1)]
                Xi = X[batch, :]
                yi = y[batch].reshape(-1)
                p_new = self._update_p(Xi)
                diff = p_new - yi
                self.bias -= learning_rate(t) * np.mean(diff) / batch_size
                self.beta -= learning_rate(t) * Xi.T @ diff / batch_size

            self.printprogress((t + 1) / self.epochs)

        return None  # self.beta is now updated and predict() can be called

    def predict(self, Xt, binary=False):
        """
        Provide a prediction for some data X.

        Parameters
        ----------
        Xt : array, shape(i, features)
            Matrix which is desired an output prediction for.

        Returns
        -------
        y_pred : array, shape(i, )
            Corresponding prediction for X.
        """
        if Xt.shape[1] != self.features:
            raise ValueError(f"Xt must have {self.features} columns!")
        y_pred = self._update_p(Xt)
        if binary:
            y_pred[y_pred < 0.5] = 0
            y_pred[y_pred >= 0.5] = 1
        return y_pred

    def _update_p(self, Xi):
        """
        Update the probability vector p, which must be done each time
        a new mini batch is picked. This method is automatically
        called within the fit() method

        Parameters
        ----------
        Xi : array, shape(i, features)
            Mini batch part of matrix X.

        Returns
        -------
        p_new : array, shape(i, )
            Updated probability vector according to the tanh function.
        """
        p_new = 1.0 / (1 + np.exp(-Xi @ self.beta + self.bias))  # sigmoid
        # p_new = np.tanh(Xi @ self.beta + self.bias)  # tanh
        return p_new.reshape(-1)

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
