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
    NN_classification(X_train, y_train, X_test, y_test)


def main_regression():
    # ----- Regression, Franke Function -----
    generate_franke_data(N=10000, noise=.5, test_size=.2)
    train_data = np.load("./data/franke/train_data.npz")
    test_data = np.load("./data/franke/test_data.npz")
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    NN_regression(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # main_classification()
    main_regression()
