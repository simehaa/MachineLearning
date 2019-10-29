from lib.functions import *
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
np.random.seed(1)


def main():
    """
    Info:
    Perform Ridge and Lasso regression on gene data. Try to predict
    The output which is the Duke CAD index
    (Coronary Artery Disease index).

    Notes:
    Data file "data_E1.csv" contains the 110 pasients with case:
    * The first column contains file IDs (not necessary for analysis)
    * the second column contains the y data: true CADi
    * the remaining columns contains 22283 features
    """
    # Read data from file
    X, y = read_file("data_E1.csv")
    # Scale data with both mean and std: e.g. y = y/std(y) - mean(y)
    X = scale(X)
    y = scale(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # Ridge (fit_intercept=False assumes that data is already scaled)
    clf_r = RidgeCV(fit_intercept=False).fit(X_train, y_train)
    y_pred_r = clf_r.predict(X_test)
    score_r = clf_r.score(X_test, y_test)
    # Plot and save figure
    save_fig(y_test, y_pred_r, score_r, "ridge.png")
    # Lasso (fit_intercept=False assumes that data is already scaled
    # Note: cv=5 is the default of a future version of scikit-learn
    clf_l = LassoCV(fit_intercept=False, cv=5).fit(X_train, y_train)
    y_pred_l = clf_l.predict(X_test)
    score_l = clf_l.score(X_test, y_test)
    # Plot and save figure
    save_fig(y_test, y_pred_l, score_l, "lasso.png")


if __name__ == "__main__":
    main()


# Test run
# python 3.7.3
# scikit-learn 0.21.3
# numpy 1.17.2
# pandas 0.25.1
# matplotlib 3.1.1
"""
$ python main.py
The file "./figures/ridge.png" already exists, do you wish to overwrite it [y/n]?
y
Figure was overwritten.
The file "./figures/lasso.png" already exists, do you wish to overwrite it [y/n]?
y
Figure was overwritten.
"""
