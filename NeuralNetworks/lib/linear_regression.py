import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import scale
from math import floor


class LinearRegression:
    """
    Regression class which uses sklearn. Includes functions to solve:
    * Ordinary Least Squares (OLS).
    * Ridge regression.
    * Lasso regression.

    Parameters
    ----------
    X : array, shape(N x p)
        Design matrix.

    y : array, shape (N, )
        Y-data points.

    Attributes
    ----------
    X : array, shape(N x p)
        Design matrix.

    y : array, shape (N, )
        Y-data points.

    X_temp : array, shape(N x p)
        Copy of X.

    y_temp : array, shape (N, )
        Copy of y.

    p : int
        Number of features.

    beta : array, shape(p, )
        Weights after fit.


    Examples
    --------
    model = Regression(X, y)
    model.ols_fit(svd=True)
    y_pred = model.predict(X)
    MSE_kfold, R2 = model.k_fold_cross_validation(10, "ols", svd=True)
    MSE_train = model.mean_squared_error(y, y_pred)
    """

    def __init__(self, X, y):
        # store all of X, y
        self.X = X
        self.y = y
        # copies of X, y (NB: these are used in ols/ridge/lasso), convenient
        # for the k-fold cross validation
        self.X_temp = X
        self.y_temp = y
        self.p = None


    def update_X(self, X):
        self.X = X
        self.X_temp = X
        return None


    def update_y(self, y):
        self.y = y
        self.y_temp = y
        return None


    def svd_inv(self, A):
        """Invert matrix A by using Singular Value Decomposition"""
        U, D, VT = np.linalg.svd(A)
        return VT.T @ np.linalg.inv(np.diag(D)) @ U.T


    def ols_fit(self, svd=False):
        """
        Find the coefficients of beta: An array of shape (p, 1), where p is the
        number of features. Beta is calculated using the X, y attributes of the
        instance.

        Parameter
        ---------
        svd : bool
            If True, (X^T X) is inverted by using SVD.
        """
        XTX = self.X_temp.T @ self.X_temp
        if svd:
            XTX_inv = self.svd_inv(XTX)
        else:
            XTX_inv = np.linalg.inv(XTX)
        self.beta = XTX_inv @ self.X_temp.T @ self.y_temp
        self.p = self.beta.shape[0]
        return None


    def ridge_fit(self, alpha=1e-6):
        """
        Find the coefficients of beta: An array of shape (p, 1), where p is the
        number of features. Beta is calculated using the X, y attributes of the
        instance.

        Parameter
        ---------
        alpha : float
            Hyperparameter for the Ridge regression
        """
        model = Ridge(alpha=alpha, normalize=True)
        model.fit(self.X_temp,self.y_temp)
        p = self.X_temp.shape[1]
        self.beta = np.transpose(model.coef_)
        self.beta[0] = model.intercept_
        self.p = self.beta.shape[0]
        return None


    def lasso_fit(self, alpha=1e-6):
        """
        Find the coefficients of beta: An array of shape (p, 1), where p is the
        number of features. Beta is calculated using the X, y attributes of the
        instance.

        Parameter
        ---------
        alpha : float
            Hyperparameter for the LASSO regression.
        """
        model = Lasso(alpha=alpha, normalize=True, tol=0.05, max_iter=2500)
        model.fit(self.X_temp,self.y_temp)
        p = self.X_temp.shape[1]
        self.beta = np.transpose(model.coef_)
        self.beta[0] = model.intercept_
        self.p = self.beta.shape[0]
        return None


    def predict(self, X):
        """
        This method can only be called after ols/ridge/lasso_regression() has
        been called. It will predict y, given X.

        Parameter
        ---------
        X : array, shape(N, p)
            Matrix to provide prediction for.

        Returns
        -------
        y_pred : array, shape(N, )
            Prediction.
        """
        if self.p:
            if X.shape[1] != self.p:
                raise ValueError(f"Model has produced a beta with {self.p} features" +
                f" and X in predict(X) has {X.shape[1]} columns.")
            y_pred = X @ self.beta
            return y_pred
        else:
            print("Warning, cannot predict because nothing has been fitted yet!" +
             " Try using ols_fit(), ridge_fit() or lasso_fit() first.")



    def mean_squared_error(self, y, y_pred):
        """Evaluate the mean squared error for y, y_pred"""
        mse = np.mean((y - y_pred)**2)
        return mse


    def r2_score(self, y, y_pred):
        """Evaluate the R2 (R squared) score for y, y_pred"""
        y_mean = np.mean(y)
        RSS = np.sum((y - y_pred)**2) # residual sum of squares
        TSS = np.sum((y - y_mean)**2) # total sum of squares
        r2 = 1 - RSS/TSS
        return r2


    def k_fold_cross_validation(self, k, method, alpha=1e-3, svd=False):
        """
        Perform the k-fold cross validation and evaluate the mean squared
        error and the R squared score.

        Parameters
        ----------
        k : int
            Number of cross validation sub sets.

        method : string
            Must be "ols", "ridge" or "lasso".

        alpha : float
            Parameter for ridge/lasso, can be ignored for ols.

        svd : bool
            Option for ols, if True, invert (X^T X) by using SVD.

        Returns
        -------
        MSE : float
            The Mean squared error.

        R2 : float
            The R^2 score.
        """
        mse = np.zeros(k)
        r2 = np.zeros(k)
        N = self.X.shape[0]
        p = np.random.permutation(N) # permutation array for shuffling of data
        length = floor(N/k) # number of indices per interval k.
        for i in range(k):
            start = i*length
            stop = (i+1)*length
            # split
            X_test = self.X[p[start:stop]]
            y_test = self.y[p[start:stop]]
            self.X_temp = np.concatenate((self.X[p[:start]],self.X[p[stop:]]),axis=0)
            self.y_temp = np.concatenate((self.y[p[:start]],self.y[p[stop:]]))
            # fit
            if method == "ols":
                self.ols_fit(svd=svd)
            elif method == "ridge":
                self.ridge_fit(alpha=alpha)
            elif method == "lasso":
                self.lasso_fit(alpha=alpha)
            else:
                raise ValueError("method must be \"osl\"/\"lasso\"/\"ridge\".")
            # predict
            y_pred = self.predict(X_test)
            # evaluate
            mse[i] = self.mean_squared_error(y_test, y_pred)
            r2[i] = self.r2_score(y_test, y_pred)

        # Reset temporary arrays
        self.X_temp = self.X
        self.y_temp = self.y
        # Evaluate mean
        MSE = np.mean(mse)
        R2 = np.mean(r2)
        return MSE, R2
