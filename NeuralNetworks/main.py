import numpy as np
from lib.preprocessing import *

# from lib.neural_network import *
from lib.logistic_regression import *


def main():
    # preprocess_raw_data() # Saves new data files in ./data/processed/

    train_data = np.load("./data/processed/train_data.npz", allow_pickle=True)
    test_data = np.load("./data/processed/test_data.npz", allow_pickle=True)
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    # SGD with mini batches
    clf = SGDClassification(batch_size=300, epochs=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc, ones, zeros = accuracy(y_test, y_pred)
    print(f"SGD accuracy: {acc*100:2.2f} %.")
    print(f"\tNo. of ones: {ones}")
    print(f"\tNo. of zeros: {zeros}")


def accuracy(y, y_pred):
    """
    Compute accuracy: fraction of how many elements in y and y_pred that
    Correspond divided by total number of elements.

    Parameters
    ----------
    y : array, shape(features, )
        Test data binary output values (0 or 1).

    y_pred : array, shape(features, )
        Predicted binary output values (0 or 1).

    Returns
    -------
    acc : float
        Accuracy score.
    """
    total = np.ones(y.shape, dtype=np.int32)
    acc = np.sum(total[np.abs(y_pred - y) < 1e-5]) / np.sum(total)
    ones = np.sum(total[y_pred == 1])
    zeros = np.sum(total[y_pred == 0])
    return acc, ones, zeros


if __name__ == "__main__":
    main()
