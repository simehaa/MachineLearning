import numpy as np
from lib.functions import *
from lib.test_functions import *
# from lib.preprocessing import *
np.random.seed(13)


def main():
    # ----- Classification, Credit Card Data -----

    # preprocess_raw_data(test_size=0.2) # Saves new data files in ./data/processed/
    train_data = np.load("./data/processed/train_data.npz")
    test_data = np.load("./data/processed/test_data.npz")
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    # LogReg(X_train, y_train, X_test, y_test, epochs=1000)
    # NN_classification(X_train, y_train, X_test, y_test)
    NN_classification_grid_search(X_train, y_train, X_test, y_test) # Best: eta=0.1 reg=1e-5


    # ----- Regression, Franke Function -----

    X_train, y_train = generate_franke_data(N=30000, noise=.3)
    X_test, y_test = generate_franke_data(N=10000, noise=.3)
    # LinReg(X_train, y_train, X_test, y_test)
    # NN_regression(X_train, y_train, X_test, y_test)
    # Animations
    # animate_franke(X_train, y_train, X_test, y_test, epochs=100, activation="sigmoid")
    # animate_franke(X_train, y_train, X_test, y_test, epochs=100, activation="tanh")
    # animate_franke(X_train, y_train, X_test, y_test, epochs=100, activation="relu")
    # animate_franke(X_train, y_train, X_test, y_test, epochs=100, activation="relu6")
    # animate_franke(X_train, y_train, X_test, y_test, epochs=100, activation="l_relu")


if __name__ == "__main__":
    # tests()
    main()
