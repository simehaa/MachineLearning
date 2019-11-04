import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from lib.preprocessing import *
from lib.neural_network import *
from lib.logistic_regression import *
from lib.functions import *


def main():
    # ----- Classification, Credit Card Data -----
    preprocess_raw_data(test_size=0.33) # Saves new data files in ./data/processed/
    train_data = np.load("./data/processed/train_data.npz")
    test_data = np.load("./data/processed/test_data.npz")
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    LogReg(X_train, y_train, X_test, y_test, epochs=100)
    # NN_classification(X_train, y_train, X_test, y_test)

    # ----- Regression, Franke Function -----
    # X_train, y_train = generate_franke_data(N=40000)
    # l = np.linspace(0,1,100)
    # X_test = np.column_stack((l, l))
    # y_test = franke_function(np.meshgrid(l, l))


def LogReg(X_train, y_train, X_test, y_test, epochs):
    print("Logistic Regression\n")

    # my own Log Reg
    clf = SGDClassification(batch_size=50, epochs=epochs)
    clf.fit(X_train, y_train)
    print("\n\t\t Error rate\t       | Area ratio")
    print("\t\t Training | Validation | Training | Validation")
    y_pred = clf.predict(X_test, binary=True)
    test_acc = np.sum(y_pred != y_test) / len(y_pred)
    test_area = roc_curve(y_test, y_pred, show=True)
    y_pred = clf.predict(X_train, binary=True)
    train_acc = np.sum(y_pred != y_train) / len(y_pred)
    train_area = roc_curve(y_train, y_pred)
    print(f"\tMy SGD  | {train_acc:2.2f}    | {test_acc:2.2f}", end='')
    print(f"       | {train_area:2.2f}     | {test_area:2.2f}")

    # Sklearn Log Reg
    clf = SGDClassifier(
        loss="log",
        penalty="none",
        fit_intercept=True,
        max_iter=epochs,
        shuffle=True,
        learning_rate="optimal",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = np.sum(y_pred != y_test) / len(y_pred)
    test_area = roc_curve(y_test, y_pred)
    y_pred = clf.predict(X_train)
    train_acc = np.sum(y_pred != y_train) / len(y_pred)
    train_area = roc_curve(y_train, y_pred)
    print(f"\tSklearn | {train_acc:2.2f}    | {test_acc:2.2f}", end='')
    print(f"       | {train_area:2.2f}     | {test_area:2.2f}")

    """
    Logistic Regression

	[##############################] 100.00 % Done.

    		 Error rate	           | Area ratio
    		 Training | Validation | Training | Validation
	My SGD  | 0.29    | 0.22       | 0.35     | 0.40
	Sklearn | 0.30    | 0.24       | 0.35     | 0.40
    """


def NN_classification(X_train, y_train, X_test, y_test):
    epochs = 12
    batch_size = 100
    learning_rate = 0.0001
    reg = 1e-4
    layers = [29, 75, 50, 34, 22, 15, 10, 1]
    cost = CrossEntropy()
    print("Neural Network\n")
    NN = NeuralNetwork()
    # EITHER do the next line OR simply load previous saved wights/biases
    NN.SGD(
        X_train,
        y_train,
        layers=layers,
        cost=cost,
        epochs=epochs,
        batch_size=batch_size,
        eta=0.01,
        reg=reg,
    )
    # NN.load_weights_and_biases(layers)

    plt.plot(np.linspace(1, epochs, epochs), NN.vali_accuracy)
    plt.show()

    # Print results:
    print("\n\tTraining set:")
    area_ratio = roc_curve(y_train, NN.predict(X_train, binary=False))
    print(f"\tArea ratio = {area_ratio:2.2f}")
    y_pred = NN.predict(X_train, binary=True)
    co, cz, to, tz = accuracy(y_train, y_pred)
    print_accuracy(co, cz, to, tz)

    print("\tTest set:")
    area_ratio = roc_curve(y_test, NN.predict(X_test, binary=False))
    print(f"\tArea ratio = {area_ratio:2.2f}")
    y_pred = NN.predict(X_test, binary=True)
    co, cz, to, tz = accuracy(y_test, y_pred)
    print_accuracy(co, cz, to, tz)

    NN.save_weights_and_biases()


def plot_confusion(y, y_pred):
    ax = skplt.metrics.plot_confusion_matrix(y, y_pred, normalize=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


if __name__ == "__main__":
    main()
