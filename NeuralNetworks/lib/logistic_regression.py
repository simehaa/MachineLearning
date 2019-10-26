import numpy as np


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
    clf.predict(X, y)
    y_pred = clf.fit(X_test)
    """

    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y, beta0=None):
        """
        Perform SGD with mini batches on X, y as training data. if initial
        guess, beta0 is not provided, then this will be a random guess with
        values in [- 0.7, + 0.7].

        Parameters
        ----------
        X : array, shape (N, features)
            Training data feature matrix.

        y : array, shape (N, )
            Training data output values.

        beta0 : array, shape(features, )
            Initial guess for parameter vector, size num_features.

        Returns
        -------
        None
            features and beta are now new attributes and predict()
            can be called.
        """
        N, self.features = X.shape
        if beta0 is not None:
            self.beta = (np.random.random(p) - 0.5) * 1.4
        else:
            if beta0.shape[0] == self.features:
                self.beta = beta0
            else:
                raise ValueError(
                    "beta0 must be of the same shape as number of columns in X"
                )

        num_batches = int(N / self.batch_size)
        batches = np.random.permutation(N)
        batches.reshape(num_batches, self.batch_size)
        learning_rate = lambda t: 1 / (t + 10)

        # iterative loop over epochs
        for t in self.epochs:
            # loop over mini batches
            for _batch in range(num_batches):
                # pick a random batch, num_batches times
                random_batch_number = np.random.randint(num_batches)
                i = batches[random_batch_number, :]  # random batch indices
                p_new = self._update_p(X[i, :])
                self.beta -= learning_rate(t) * X[i, :].T @ (y[i] - p_new)

        return None  # self.beta is now updated and predict() can be called

    def predict(self, Xt):
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
        else:
            y_pred = Xt @ self.beta
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
            Updated probability vector according to the sigmoid function.
        """
        p_new = 1.0 / (1 + np.exp(-Xi @ self.beta))
        return p_new