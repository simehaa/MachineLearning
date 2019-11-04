import numpy as np
import matplotlib.pyplot as plt
from scikitplot.helpers import cumulative_gain_curve
from sklearn.metrics import auc


def generate_franke_data(N=10000, noise=0.5, test_size=0.2):
    x1 = np.random.uniform(0, 1, N)
    x2 = np.random.uniform(0, 1, N)
    X = np.column_stack((x1, x2))
    y = franke_function(x1, x2) + np.random.normal(0, noise, N)
    return X, y


def franke_function(x, y):
    """
    The Franke function f(x, y). The inputs are elements or vectors with
    elements in the domain of [0, 1].
    """
    if np.shape(x) != np.shape(y):
        raise ValueError("x and y must be of same shape!")

    term = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term += 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term += 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term += -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term


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


def roc_curve(
    y_true,
    y_probas,
    title="Cumulative Gains Curve",
    title_fontsize="large",
    text_fontsize="medium",
    ax=None,
    figsize=None,
    show=False,
):
    """
    Plot the ROC-curve diagrams
    Modified code from scikitplot.helpers.plot_cumulative_gain
    """

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            "Cannot calculate Cumulative Gains for data with "
            "{} category/ies".format(len(classes))
        )

    y_probas = y_probas.reshape((len(y_probas), 1))
    y_probas = np.concatenate((np.zeros((len(y_probas), 1)), y_probas), axis=1)

    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])
    ones = np.sum(y_true)
    tot = len(y_true)
    best_curve_x = [0, ones, tot]
    best_curve_y = [0, ones, ones]
    bottom_area = tot*ones / 2
    best_curve_area = auc(best_curve_x, best_curve_y) - bottom_area
    model_curve_area = auc(percentages*tot, gains2*ones) - bottom_area
    area_ratio = model_curve_area / best_curve_area
    if show:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(percentages*tot, gains2*ones, lw=2, label="Model")
        ax.plot(best_curve_x, best_curve_y, lw=2, label="Best curve")
        ax.set_xlim([0.0, tot*1.0])
        ax.set_ylim([0.0, ones*1.2])
        ax.plot([0, tot], [0, ones], "k--", lw=2, label="Baseline")
        ax.set_xlabel("Number of total data", fontsize=text_fontsize)
        ax.set_ylabel("Cumulative number of target data", fontsize=text_fontsize)
        ax.tick_params(labelsize=text_fontsize)
        ax.grid()
        ax.legend(loc=4, fontsize=text_fontsize)
        plt.show()

    return area_ratio
