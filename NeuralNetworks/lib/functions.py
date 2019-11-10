# Dependencies: numpy, matplotlib, sklearn, tensorflow and seaborn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from scikitplot.helpers import cumulative_gain_curve
import seaborn as sns
import tensorflow as tf
# other directories in this project
from lib.neural_network import *
from lib.logistic_regression import *
from lib.linear_regression import *


def LogReg(X_train, y_train, X_test, y_test, epochs, sklrn=False):
    """
    Perform logistic regression on the data given as parameters. Uses the
    SGDClassification class in logistic_regression.py.

    Parameters
    ----------
    X_train : array, shape(N, p)
        Training data.

    y_train : array, shape(N, )
        Training data.

    X_test : array, shape(N, p)
        Testing data.

    y_test : array, shape(N, )
        Testing data.

    epochs : int
        Number of epochs to run the algorithm for.

    sklrn : bool
        If true, runs scikit-learn's SGDClassifier in addition to double
        check the results.

    Returns
    -------
    None : prints/plots the results directly.
    """
    print("Logistic Regression\n")

    # my own Log Reg
    clf = SGDClassification(batch_size=100, epochs=epochs)
    clf.fit(X_train, y_train)
    print("\n\t\t Error rate\t       | Area ratio\t")
    print("\t\t Training | Validation | Training | Validation")
    y_pred = clf.predict(X_test, binary=True)
    test_acc = np.sum(y_pred != y_test) / len(y_pred)
    test_area = roc_curve(y_test, y_pred, show=True)
    y_pred = clf.predict(X_train, binary=True)
    train_acc = np.sum(y_pred != y_train) / len(y_pred)
    train_area = roc_curve(y_train, y_pred)
    print(f"\tMy SGD  | {train_acc:2.2f}    | {test_acc:2.2}", end="")
    print(f"       | {train_area:2.2}     | {test_area:2.2}")

    sorting_smoothing_method(clf.predict(X_test, binary=False), y_test)

    # Sklearn Log Reg
    if sklrn:
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
        print(f"\tSklearn | {train_acc:2.2f}    | {test_acc:2.2}", end="")
        print(f"       | {train_area:2.2}     | {test_area:2.2}")

    """
    Logistic Regression

	[##############################] 100.00 % Done.

    		 Error rate	           | Area ratio
    		 Training | Validation | Training | Validation
	My SGD  | 0.29    | 0.22       | 0.35     | 0.40
	Sklearn | 0.30    | 0.24       | 0.35     | 0.40
    """


def sorting_smoothing_method(y_pred, y_test, n=50):
    """
    Esimation of actual probability of default by SSM.

    Parameters
    ----------
    y_pred : array, shape(N, )
        Prediction array with continuous values in the range [0, 1].

    y_test : array, shape(N, )
        Output value array with binary values 0 or 1.

    n : int
        Size of neighbothood of points to use in SSM (number of points for
        smoothing will be from i-n, ..., i+n)

    Returns
    -------
    None : creates a plot.
    """
    # order data from min to max
    ind = np.argsort(y_pred)
    y_pred = y_pred[ind]
    y_binary = y_test[ind]

    # predicted probability
    pred_prob = y_pred[n:-n]
    N = y_pred.shape[0]
    est_prob = np.zeros(N - n - n)

    # estimated probability according to SSM
    for i in range(50, N-50):
        est_prob[i - 50] = np.mean(y_binary[i - n: i + n])

    # linear regression
    X = np.column_stack((np.ones(N - n - n), pred_prob))
    LinReg = LinearRegression(X, est_prob)
    LinReg.ols_fit()
    line = LinReg.predict(X)
    intercept, slope = LinReg.beta
    mse, r2 = LinReg.k_fold_cross_validation(5, "ols")
    print(f"\tR2 score = {r2:1.3f}")

    # plot
    plt.plot(pred_prob, line, 'k-',
        label=f"y = {slope:1.3f}x + " +
        f"{intercept:1.4f}\n" +
        f"$R^2$ = {r2:1.3f}")
    plt.plot(pred_prob, est_prob, 'k.', alpha=.1)
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Estimated Actual Probability")
    plt.legend()
    plt.show()


def NN_classification_grid_search(X_train, y_train, X_test, y_test):
    """
    Perform classification by using a Neural Network. The layers and
    activation function can be specified under.

    Parameters
    ----------
    X_train : array, shape(N, p)
        Training data.

    y_train : array, shape(N, )
        Training data.

    X_test : array, shape(N, p)
        Testing data.

    y_test : array, shape(N, )
        Testing data.

    Returns
    -------
    None : prints/plots the results directly.
    """
    print("Neural Network\n")
    layers = [41, 100, 75, 50, 34, 1]
    cost = CrossEntropy()
    act_fns = ["sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"]
    NN = NeuralNetwork(layers=layers, cost=cost, act_fns=act_fns)

    epochs = 10
    batch_size = 100
    learning_rates = np.logspace(-3, 0, 4)
    regular_params = np.logspace(-1, -6, 6)
    area_ratios_tr = np.zeros((4, 6))
    area_ratios_te = np.zeros((4, 6))

    for i, eta in enumerate(learning_rates):
        for j, reg in enumerate(regular_params):
            NN.SGD(
                X_train, y_train, validation_data=(X_test, y_test),
                epochs=epochs, batch_size=batch_size, eta=eta, reg=reg
            )
            # y_pred = NN.predict(X_test, binary=False)
            # sortin_smoothing_method(y_pred, y_test)
            area_ratio_tr = roc_curve(y_train, NN.predict(X_train, binary=False))
            area_ratio_te = roc_curve(y_test, NN.predict(X_test, binary=False))
            area_ratios_tr[i, j] = area_ratio_tr
            area_ratios_te[i, j] = area_ratio_te

    def plot_heat_map(area_ratios, title):
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(
            area_ratios_tr,
            ax=ax,
            square=True,
            annot=True,
            cmap="YlGnBu",
            xticklabels=[f"{i:1g}" for i in regular_params],
            yticklabels=[f"{i:1g}" for i in learning_rates],
        )
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.title(title)
        plt.xlabel("L2 Regularization parameter")
        plt.ylabel("Learning rate")
        plt.show()

    plot_heat_map(area_ratios_tr, "Area ratios from training set")
    plot_heat_map(area_ratios_te, "Area ratios from test set")

    eta_i, reg_i = np.unravel_index(
        np.argmax(area_ratios_tr, axis=None), area_ratios_te.shape
    )
    print("\tTraining data:")
    print(f"\t\tBest area ratio = {area_ratios_tr[eta_i, reg_i]:2.2}")
    print(f"\t\teta = {learning_rates[eta_i]:g}, reg = {regular_params[reg_i]:g}")
    eta_i, reg_i = np.unravel_index(
        np.argmax(area_ratios_te, axis=None), area_ratios_te.shape
    )
    print("\tTest data:")
    print(f"\t\tBest area ratio = {area_ratios_tr[eta_i, reg_i]:2.2}")
    print(f"\t\teta = {learning_rates[eta_i]:g}, reg = {regular_params[reg_i]:g}")


def NN_classification(X_train, y_train, X_test, y_test):
    """
    Perform classification by using a Neural Network. The layers and
    activation function can be specified under.

    Parameters
    ----------
    X_train : array, shape(N, p)
        Training data.

    y_train : array, shape(N, )
        Training data.

    X_test : array, shape(N, p)
        Testing data.

    y_test : array, shape(N, )
        Testing data.

    Returns
    -------
    None : prints/plots the results directly.
    """
    print("Neural Network\n")
    layers = [41, 100, 75, 50, 34, 1]
    cost = CrossEntropy()
    act_fns = ["sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"]
    NN = NeuralNetwork(layers=layers, cost=cost, act_fns=act_fns)

    epochs = 10
    batch_size = 100
    eta = 0.1
    reg = 1e-5
    NN.SGD(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size, eta=eta, reg=reg
    )
    # Error rates
    y_pred = NN.predict(X_train, binary=True)
    err_tr = 1 - np.sum(y_pred == y_train) / len(y_train)
    y_pred = NN.predict(X_test, binary=True)
    err_te = 1 - np.sum(y_pred == y_test) / len(y_test)

    y_pred = NN.predict(X_train, binary=False)
    area_ratio_tr = roc_curve(y_train, y_pred, show=False)
    y_pred = NN.predict(X_test, binary=False)
    area_ratio_te = roc_curve(y_test, y_pred, show=True)
    sorting_smoothing_method(y_pred, y_test)

    print("\tTraining data:")
    print(f"\t\tBest err rate = {err_tr:2.2}, area ratio = {area_ratio_tr:2.2}")
    print("\tTest data:")
    print(f"\t\tBest err rate = {err_te:2.2}, area ratio = {area_ratio_te:2.2}")
    """
    Evaluating error rates and creating ROC curve
	Training Neural Network with 10 epochs
	[##############################] 100.00 % Done.
	Training data:
		Best err rate = 0.29, area ratio = 0.55
	Test data:
		Best err rate = 0.21, area ratio = 0.55
    """


def NN_regression(X_train, y_train, X_test, y_test):
    """
    Perform regression on the data given as parameters by using a neural
    network.

    Parameters
    ----------
    X_train : array, shape(N, p)
        Training data.

    y_train : array, shape(N, )
        Training data.

    X_test : array, shape(N, p)
        Testing data.

    y_test : array, shape(N, )
        Testing data.

    Returns
    -------
    None : prints/plots the results directly.
    """
    cost = MSE()
    layers = [2, 100, 60, 1]
    act_fns = ["tanh", "tanh", "linear"]
    NN = NeuralNetwork(layers=layers, cost=cost, act_fns=act_fns)
    epochs = 750
    batch_size = 100
    learning_rates = np.logspace(-2, -4, 3)
    regular_params = np.logspace(-4, -1, 4)
    r2_scores = np.zeros((3, 4))
    epoch_arr = np.linspace(1, epochs, epochs)

    # MSE vs. epochs plot:
    fig = plt.figure()
    ax = plt.subplot(111)
    line = [["b--", "b-.", "b:"],
             ["g--", "g-.", "g:"],
             ["r--", "r-.", "r:"],
             ["y--", "y-.", "y:"]
    ]

    for i, eta in enumerate(learning_rates):
        for j, reg in enumerate(regular_params):
            NN.SGD(
                X_train, y_train, validation_data=(X_test, y_test),
                epochs=epochs, batch_size=batch_size, eta=eta, reg=reg
            )
            r2_scores[i, j] = r2_score(y_test, NN.predict(X_test))
            ind = np.argmin(NN.cost_arr)
            e = epoch_arr[ind]
            c = NN.cost_arr[ind]
            plt.plot(epoch_arr, NN.cost_arr, line[j][i], label=rf"$\eta, \lambda=$ ({eta:1.0e},{reg:1.0e})")
            plt.plot(e, c, 'ro')

    plt.plot(epoch_arr, np.ones(epoch_arr.shape)*0.09, "k-", label=r"Irreducible error $\sigma^2$")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow=True)
    plt.ylim(0.089, 0.115)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(
        r2_scores,
        square=True,
        annot=True,
        cmap="YlGnBu",
        xticklabels=[f"{i:1g}" for i in regular_params],
        yticklabels=[f"{i:1g}" for i in learning_rates],
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(r"$R^2$ scores")
    plt.xlabel("L2 Regularization parameter")
    plt.ylabel("Learning rate")
    plt.show()

    # Plot of franke function and prediction on mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("y")
    ax.set_zlim(-0.5, 1.5)
    # predict a mesh of data
    l = np.linspace(0, 1, 101)
    x1_mesh, x2_mesh = np.meshgrid(l, l)
    x1_flat, x2_flat = x1_mesh.flatten(), x2_mesh.flatten()
    y_pred = NN.predict(np.column_stack((x1_flat, x2_flat)))
    y_pred_mesh = np.reshape(y_pred, x1_mesh.shape)
    func_mesh = franke_function(x1_flat, x2_flat).reshape(x1_mesh.shape)
    surface_pred = ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, cmap=mpl.cm.coolwarm, alpha=.7)
    surface_true = ax.plot_surface(x1_mesh, x2_mesh, func_mesh, alpha=.3)
    fig.colorbar(surface_pred, shrink=0.5)
    plt.show()


def LinReg(X_train, y_train, X_test, y_test):
    """
    Perform linear regression on the data given as parameters by using a
    Ridge regression and 5-fold CV.

    Parameters
    ----------
    X_train : array, shape(N, p)
        Training data.

    y_train : array, shape(N, )
        Training data.

    X_test : array, shape(N, p)
        Testing data.

    y_test : array, shape(N, )
        Testing data.

    Returns
    -------
    None : prints/plots the results directly.
    """
    regular_params = np.linspace(-13, -5, 9)
    poly_degrees = np.linspace(7, 15, 9, dtype=np.uint32)
    r2_scores = np.zeros((9, 9))

    for i, reg in enumerate(regular_params):
        for j, deg in enumerate(poly_degrees):
            X = create_polynomial_design_matrix(X_train, deg)
            Reg = LinearRegression(X, y_train)
            mse, r2 = Reg.k_fold_cross_validation(5, "ridge", alpha=10**reg)
            r2_scores[i, j] = r2

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(
        r2_scores,
        square=True,
        annot=True,
        cmap="YlGnBu",
        xticklabels=[f"{i:1g}" for i in poly_degrees],
        yticklabels=[f"{i:1g}" for i in regular_params],
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(r"$R^2$ scores")
    plt.xlabel("Polynomial degree")
    plt.ylabel("log10 (Regularization parameter)")
    plt.show()


def create_polynomial_design_matrix(X_data, degree):
    """
    X_data = [x_data  y_data]
    Create polynomial design matrix on the form where columns are:
    X = [1  x  y  x**2  xy  y**2  x**3  x**2y  ... ]
    """
    X = PolynomialFeatures(degree).fit_transform(X_data)
    return X


def r2_score(y, y_pred):
    """Evaluate the R2 (R squared) score for y, y_pred"""
    y_mean = np.mean(y)
    RSS = np.sum((y - y_pred)**2) # residual sum of squares
    TSS = np.sum((y - y_mean)**2) # total sum of squares
    r2 = 1 - RSS/TSS
    return r2


def plot_confusion(y, y_pred):
    """Plot the confusion matrix of y and y_pred"""
    ax = skplt.metrics.plot_confusion_matrix(y, y_pred, normalize=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


def generate_franke_data(N=10000, noise=0.5):
    """Generate a data set of N points with noisy franke function data"""
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
    """
    Print accuracy in terms of both correct ones and zeros and
    total accuracy
    """
    acc = (correct_ones + correct_zeros) / (total_ones + total_zeros)
    print(f"\tCorrect ones  : {correct_ones} / {total_ones}")
    print(f"\tCorrect zeros : {correct_zeros} / {total_zeros}")
    print(f"\tOverall accuracy : {acc*100:2.2f} %\n")


def roc_curve(
    y_true,
    y_probas,
    title_fontsize="large",
    text_fontsize="medium",
    figsize=None,
    show=False,
):
    """
    Plot the ROC-curve diagrams
    Modified code from scikitplot.helpers.plot_cumulative_gain

    Parameters
    ----------
    y_true : array, shape(N, )
        Binary output values

    y_probas : array, shape(N, )
        Continuous predicted probabilities.

    title_fontsize : str
        Argument for plot.

    text_fontsize : str
        Argument for plot.

    figsize : str
        Argument for plot.

    show : bool
        If true, plots the figure, if False, only returns the area ratio

    Returns
    -------
    area_ratio : float
        Calculated area ratio in cumulative gain chart.
    """

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            "Cannot calculate Cumulative Gains for data with "
            "{} category/ies".format(len(classes))
        )

    y_probas = y_probas.flatten()
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas, classes[1])
    ones = np.sum(y_true)
    tot = len(y_true)
    best_curve_x = [0, ones, tot]
    best_curve_y = [0, ones, ones]
    base_curve_x = [0, tot]
    base_curve_y = [0, ones]
    bottom_area = auc(base_curve_x, base_curve_y)
    best_curve_area = auc(best_curve_x, best_curve_y) - bottom_area
    model_curve_area = auc(percentages * tot, gains2 * ones) - bottom_area
    area_ratio = model_curve_area / best_curve_area
    if show:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(percentages * tot, gains2 * ones, lw=2, label="Model")
        ax.plot(best_curve_x, best_curve_y, lw=2, label="Best curve")
        ax.plot(base_curve_x, base_curve_y, "k--", lw=2, label="Baseline")
        ax.set_xlim([0.0, tot * 1.0])
        ax.set_ylim([0.0, ones * 1.2])
        ax.set_xlabel("Number of total data", fontsize=text_fontsize)
        ax.set_ylabel("Cumulative number of target data", fontsize=text_fontsize)
        ax.tick_params(labelsize=text_fontsize)
        ax.grid()
        ax.legend(loc=4, fontsize=text_fontsize)
        plt.show()

    return area_ratio


def animate_franke(X_train, y_train, X_test, y_test, epochs=10, activation="sigmoid"):
    """
    Animate the time (epoch) evolution of a neural network regression of the
    franke function.

    Parameters
    ----------
    X_train : array, shape(N, p)
        Training data.

    y_train : array, shape(N, )
        Training data.

    X_test : array, shape(N, p)
        Testing data.

    y_test : array, shape(N, )
        Testing data.

    epochs : int
        Total number of epochs (also equivalent to the number of frames in
        output gif).

    activation : str
        Which activation function to use between the hidden layers.

    Returns
    -------
    None : Produces a gif.
    """
    size = 50
    l = np.linspace(0.01, 0.99, size)
    x1_mesh, x2_mesh = np.meshgrid(l, l)
    x1_flat, x2_flat = x1_mesh.flatten(), x2_mesh.flatten()
    X_test = np.column_stack((x1_flat, x2_flat))
    y_test = franke_function(x1_flat, x2_flat)

    # NN setup
    NN = NeuralNetwork(
        layers=[2, 100, 60, 1], cost=MSE(),
        act_fns=[activation, activation, "linear"]
    )
    NN.SGD(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=epochs, batch_size=100, eta=1e-2, reg=1e-2,
        save_frames=True
    )
    cost_arr = NN.cost_arr
    # First frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Epoch {0:3}/{epochs}, MSE = {cost_arr[0]:5g}")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("y")
    ax.set_zlim(-0.5, 1.5)
    # predict a mesh of data
    plot_args = {'rstride': 1, 'cstride': 1, 'color': "k",
                 'linewidth': 0.01, 'antialiased': True,
                 'shade': True, 'alpha': .35, 'zorder': 0.5
                 }
    y_pred = np.load("./data/frames/frame0.npz")["y"]
    y_pred_mesh = np.reshape(y_pred, x1_mesh.shape)
    func = franke_function(x1_flat, x2_flat).reshape(size, size)
    plot = ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, **plot_args)
    plot2 = ax.plot_surface(x1_mesh, x2_mesh, func, alpha=.45, cmap=mpl.cm.coolwarm, zorder=0.3)

    def update_surf(num, x1_mesh, x2_mesh, cost_arr):
        y_pred = np.load("./data/frames/frame" + str(num) + ".npz")["y"]
        y_pred_mesh = np.reshape(y_pred, x1_mesh.shape)
        ax.clear()
        ax.set_title(f"Epoch {num:3}/{epochs}, MSE = {cost_arr[num]:5g}")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel("y")
        ax.set_zlim(-0.5, 1.5)
        plot = ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, **plot_args)
        plot2 = ax.plot_surface(x1_mesh, x2_mesh, func, alpha=.45, cmap=mpl.cm.coolwarm, zorder=0.3)
        return plot, plot2

    ani = animation.FuncAnimation(
        fig, update_surf, epochs, fargs=(x1_mesh, x2_mesh, cost_arr), interval=200, blit=False
    )

    print("\tWriting \"franke_" + activation + ".gif\".")
    ani.save("./animations/franke_" + activation + ".gif",  fps=5,  writer='imagemagick')
