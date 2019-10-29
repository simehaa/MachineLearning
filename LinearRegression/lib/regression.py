import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import scale
from math import floor


class Regression:
    """
    Info:
    Regression class which uses sklearn. Includes functions to solve:
    * Ordinary Least Squares (OLS).
    * Ridge regression.
    * Lasso regression.

    Initialize:
    * X: (N x p) design matrix.
    * y: array containing (N x 1) data points.

    Methods:
    * update_X(X), update X, X_temp attributes
    * update_Y(Y), update Y, Y_temp attributes
    * svd_inv(A), invert A by using SVD
    * ols_fit(svd=False)         |
    * ridge_fit(alpha,svd=False) |> Saves new attributes beta, p
    * lasso_fit(alpha)           |
    * predict(X), return y_prediction (Note: can only be done after fit)
    * mean_squared_error(y, y_pred), return MSE
    * r2_score(y, y_pred), return R2
    * k_fold_cross_validation(k, method, alpha=1e-3, svd=False), apply k-fold CV

    Example:
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
        """
        Info:
        Invert matrix A by using Singular Value Decomposition

        Input:
        * A: matrix

        Output
        * A_inverted: matrix
        """
        U, D, VT = np.linalg.svd(A)
        return VT.T @ np.linalg.inv(np.diag(D)) @ U.T

    def ols_fit(self, svd=False):
        """
        Info:
        Find the coefficients of beta: An array of shape (p, 1), where p is the
        number of features. Beta is calculated using the X, y attributes of the
        instance.

        Output:
        * beta: The coefficient vector for the OLS scheme.
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
        Info:
        Find the coefficients of beta: An array of shape (p, 1), where p is the
        number of features. Beta is calculated using the X, y attributes of the
        instance.

        Input:
        * alpha: parameter for this regression type
        * svd: if True, SVD is used for matrix inversion

        Output:
        * beta: The coefficient vector for the Ridge scheme
        """
        model = Ridge(alpha=alpha, normalize=True)
        model.fit(self.X_temp, self.y_temp)
        p = self.X_temp.shape[1]
        self.beta = np.transpose(model.coef_)
        self.beta[0] = model.intercept_
        self.p = self.beta.shape[0]
        return None

    def lasso_fit(self, alpha=1e-6):
        """
        Info:
        Find the coefficients of beta: An array of shape (p, 1), where p is the
        number of features. Beta is calculated using the X, y attributes of the
        instance.

        Input:
        * alpha: parameter for this regression type

        Output:
        * beta: The coefficient vector for the Lasso scheme.
        """
        model = Lasso(alpha=alpha, normalize=True, tol=0.05, max_iter=2500)
        model.fit(self.X_temp, self.y_temp)
        p = self.X_temp.shape[1]
        self.beta = np.transpose(model.coef_)
        self.beta[0] = model.intercept_
        self.p = self.beta.shape[0]
        return None

    def predict(self, X):
        """
        Info:
        This method can only be called after ols/ridge/lasso_regression() has
        been called. It will predict y, given X.

        Input:
        * X: values of which y will be predicted.

        Output:
        * y_pred: the y prediction values.
        """
        if self.p:
            if X.shape[1] != self.p:
                raise ValueError(
                    f"Model has produced a beta with {self.p} features"
                    + f" and X in predict(X) has {X.shape[1]} columns."
                )
            y_pred = X @ self.beta
            return y_pred
        else:
            print(
                "Warning, cannot predict because nothing has been fitted yet!"
                + " Try using ols_fit(), ridge_fit() or lasso_fit() first."
            )

    def mean_squared_error(self, y, y_pred):
        """Evaluate the mean squared error for y, y_pred"""
        mse = np.mean((y - y_pred) ** 2)
        return mse

    def r2_score(self, y, y_pred):
        """Evaluate the R2 (R squared) score for y, y_pred"""
        y_mean = np.mean(y)
        RSS = np.sum((y - y_pred) ** 2)  # residual sum of squares
        TSS = np.sum((y - y_mean) ** 2)  # total sum of squares
        r2 = 1 - RSS / TSS
        return r2

    def k_fold_cross_validation(self, k, method, alpha=1e-3, svd=False):
        """
        Info:
        Perform the k-fold cross validation and evaluate the mean squared
        error and the R squared score.

        Input:
        * k
        * method: "ols", "ridge" or "lasso"
        * alpha: parameter for ridge/lasso, can be ignored for ols

        Output:
        * MSE
        * R2
        """
        mse = np.zeros(k)
        r2 = np.zeros(k)
        N = self.X.shape[0]
        p = np.random.permutation(N)  # permutation array for shuffling of data
        length = floor(N / k)  # number of indices per interval k.
        for i in range(k):
            start = i * length
            stop = (i + 1) * length
            # split
            X_test = self.X[p[start:stop]]
            y_test = self.y[p[start:stop]]
            self.X_temp = np.concatenate((self.X[p[:start]], self.X[p[stop:]]), axis=0)
            self.y_temp = np.concatenate((self.y[p[:start]], self.y[p[stop:]]))
            # fit
            if method == "ols":
                self.ols_fit(svd=svd)
            elif method == "ridge":
                self.ridge_fit(alpha=alpha)
            elif method == "lasso":
                self.lasso_fit(alpha=alpha)
            else:
                raise ValueError('method must be "osl"/"lasso"/"ridge".')
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
