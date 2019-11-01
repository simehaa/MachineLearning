import numpy as np
def LogReg_classification(X_train, y_train, X_test, y_test, runs, epochs):
    print("Logistic Regression\n")
    runs = 10
    epochs = 100
    print(f"\tAveraging over {runs} runs with {epochs} epochs")
    res = np.zeros((runs, 4))
    acc = np.zeros(runs)
    for i in range(runs):
        clf = SGDClassification(batch_size=100, epochs=epochs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        res[i, :] = accuracy(y_test, y_pred)
        acc[i] = 100 * (res[i, 0] + res[i, 1]) / (res[i, 2] + res[i, 3])
    total_ones = np.sum(y_test == 1)
    total_zeros = np.sum(y_test == 0)
    print(f"\tAverage correct ones  : {np.mean(res[:,0])} / {total_ones}")
    print(f"\tAverage correct zeros : {np.mean(res[:,1])} / {total_zeros}")
    print(
        f"\tAccuracy over {runs} runs : {np.mean(acc):2.2f} +/- {np.std(acc):2.2f} %\n"
    )


def accuracy(y_test, y_pred):
    """
    Compute accuracy: fraction of how many elements in y and y_pred that
    Correspond divided by total number of elements.

    Parameters
    ----------
    y_test : array, shape(features, )
        Test data binary output values (0 or 1).

    y_pred : array, shape(features, )
        Predicted binary output values (0 or 1).

    Returns
    -------
    Res : array, shape(4, )
        [Correct no. of ones, correct no. of zeros,
        Total no. of ones, total no. of zeros]
    """
    total_ones = np.sum(y_test == 1)
    total_zeros = np.sum(y_test == 0)
    correct_ones = np.sum(np.logical_and(y_test == y_pred, y_test == 1))
    correct_zeros = np.sum(np.logical_and(y_test == y_pred, y_test == 0))
    return correct_ones, correct_zeros, total_ones, total_zeros


def print_accuracy(correct_ones, correct_zeros, total_ones, total_zeros):
    acc = (correct_ones + correct_zeros) / (total_ones + total_zeros)
    print(f"\tCorrect ones  : {correct_ones} / {total_ones}")
    print(f"\tCorrect zeros : {correct_zeros} / {total_zeros}")
    print(f"\tOverall accuracy : {acc*100:2.2f} %\n")
