import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
# from lib.preprocessing import *
from lib.neural_network import *
from lib.logistic_regression import *
from lib.functions import *



def main():
    # preprocess_raw_data(test_size=0.2) # Saves new data files in ./data/processed/
    train_data = np.load("./data/processed/train_data.npz")
    test_data = np.load("./data/processed/test_data.npz")
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    # Neural Network Classification
    NN_classification(X_train, y_train, X_test, y_test)



    # LogReg_classification(X_train, y_train, X_test, y_test, runs=10, epochs=100)

def NN_classification(X_train, y_train, X_test, y_test):
    epochs = 50
    batch_size = 100
    learning_rate = 0.01
    cost_function = "xentropy"
    reg = 1e-6
    layers = [29, 100, 60, 40, 20, 10, 1]
    print("Neural Network\n")
    NN = NeuralNetwork()
    # EITHER do the next line OR simply load previous saved wights/biases
    # NN.set_layers(layers)
    # NN.SGD(X_train, y_train, layers=layers, epochs=epochs,
        # batch_size=batch_size, eta=0.01, cost_function=cost_function,
        # reg=reg
    # )
    NN.load_weights_and_biases(layers)

    # Print results:
    print("\n\tTraining set:")
    y_pred = NN.predict(X_train)
    co, cz, to, tz = accuracy(y_train, y_pred)
    ax = skplt.metrics.plot_confusion_matrix(y_train, y_pred, normalize=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+0.5, top-0.5)
    plt.show()
    print_accuracy(co, cz, to, tz)
    print("\tTest set:")
    y_pred = NN.predict(X_test)
    co, cz, to, tz = accuracy(y_test, y_pred)
    # skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    # plt.show()
    print_accuracy(co, cz, to, tz)
    NN.save_weights_and_biases()

if __name__ == "__main__":
    main()
