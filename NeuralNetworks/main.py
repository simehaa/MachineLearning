import numpy as np
from lib.functions import *

# from lib.preprocessing import *


def main_classification():
    # ----- Classification, Credit Card Data -----
    # preprocess_raw_data(test_size=0.2) # Saves new data files in ./data/processed/
    train_data = np.load("./data/processed/train_data.npz")
    test_data = np.load("./data/processed/test_data.npz")
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    # LogReg(X_train, y_train, X_test, y_test, epochs=100)
    # NN_classification(X_train, y_train, X_test, y_test)


def main_regression():
    # ----- Regression, Franke Function -----
    X_train, y_train = generate_franke_data(N=10000, noise=.5)
    X_test, y_test = generate_franke_data(N=5000, noise=.5)
    NN_regression(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # main_classification()
    main_regression()
