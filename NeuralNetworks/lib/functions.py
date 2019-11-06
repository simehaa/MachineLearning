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

from lib.neural_network import *
from lib.logistic_regression import *
from lib.linear_regression import *



def LogReg(X_train, y_train, X_test, y_test, epochs):
    print("Logistic Regression\n")

    # my own Log Reg
    clf = SGDClassification(batch_size=50, epochs=epochs)
    clf.fit(X_train, y_train)
    print("\n\t\t Error rate\t       | Area ratio\t")
    print("\t\t Training | Validation | Training | Validation")
    y_pred = clf.predict(X_test, binary=True)
    test_acc = np.sum(y_pred != y_test) / len(y_pred)
    test_area = roc_curve(y_test, y_pred, show=True)
    y_pred = clf.predict(X_train, binary=True)
    train_acc = np.sum(y_pred != y_train) / len(y_pred)
    train_area = roc_curve(y_train, y_pred)
    print(f"\tMy SGD  | {train_acc:2.2}    | {test_acc:2.2}", end="")
    print(f"       | {train_area:2.2}     | {test_area:2.2}")

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
    print(f"\tSklearn | {train_acc:2.2}    | {test_acc:2.2}", end="")
    print(f"       | {train_area:2.2}     | {test_area:2.2}")

    """
    Logistic Regression

	[##############################] 100.00 % Done.

    		 Error rate	           | Area ratio
    		 Training | Validation | Training | Validation
	My SGD  | 0.29    | 0.22       | 0.35     | 0.40
	Sklearn | 0.30    | 0.24       | 0.35     | 0.40
    """


def NN_classification(X_train, y_train, X_test, y_test):
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
        np.argmax(area_ratios_te, axis=None), area_ratios_te.shape
    )
    eta_te = learning_rates[eta_i]
    reg_te = regular_params[reg_i]
    area_ratio_te = area_ratios_te[eta_i, reg_i]
    area_ratio_tr = area_ratios_tr[eta_i, reg_i]

    # Lift chart on test set with best eta and reg_param:
    print("\n\tEvaluating error rates and creating ROC curve")
    NN.SGD(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size, eta=eta_te, reg=reg_te
    )
    # Error rates
    y_pred = NN.predict(X_train, binary=True)
    err_tr = 1 - np.sum(y_pred == y_train) / len(y_train)
    y_pred = NN.predict(X_test, binary=True)
    err_te = 1 - np.sum(y_pred == y_test) / len(y_test)
    # ROC curve
    y_pred = NN.predict(X_test, binary=False)
    roc_curve(y_test, y_pred, show=True)

    print("\tTraining data:")
    print(f"\t\tBest err rate = {err_tr:2.2}, area ratio = {area_ratio_tr:2.2}")
    print("\tTest data:")
    print(f"\t\tBest err rate = {err_te:2.2}, area ratio = {area_ratio_tr:2.2}")
    """
    Evaluating error rates and creating ROC curve
	Training Neural Network with 10 epochs
	[##############################] 100.00 % Done.
	Training data:
		Best err rate = 0.29, area ratio = 0.55
	Test data:
		Best err rate = 0.21, area ratio = 0.55
    """


def tensorflow(X_train, y_train, X_test, y_test):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(60, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear")
            # tf.keras.layers.Dropout(0.2),
        ]
    )  # try two outputs and softmax (well taylored for )
    model.compile(
        optimizer="adam", loss="mse", metrics=["r2"]
    )
    model.fit(
        X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test)
    )

    y_pred = model.predict(X_test)
    return y_pred


def NN_regression(X_train, y_train, X_test, y_test):
    cost = MSE()
    layers = [2, 100, 60, 1]
    act_fns = ["tanh", "tanh", "linear"]
    NN = NeuralNetwork(layers=layers, cost=cost, act_fns=act_fns)
    epochs = 10
    batch_size = 100
    learning_rates = np.logspace(-2, -3, 2)
    regular_params = np.logspace(-3, -1, 3)
    r2_scores = np.zeros((2, 3))
    epoch_arr = np.linspace(1, epochs, epochs)

    for i, eta in enumerate(learning_rates):
        for j, reg in enumerate(regular_params):
            NN.SGD(
                X_train, y_train, validation_data=(X_test, y_test),
                epochs=epochs, batch_size=batch_size, eta=eta, reg=reg
            )
            r2_scores[i, j] = r2_score(y_test, NN.predict(X_test))
            plt.plot(epoch_arr, NN.cost_arr, label=rf"$\eta=${eta:g}, $\lambda=${reg:g}")

    plt.ylabel("Cost function (MSE)")
    plt.xlabel("Epochs")
    plt.grid()
    plt.legend(loc="best")
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
    ax = skplt.metrics.plot_confusion_matrix(y, y_pred, normalize=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


def generate_franke_data(N=10000, noise=0.5):
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


def animate_franke(X_train, y_train, X_test, y_test, epochs=10):
    plot_args = {'rstride': 1, 'cstride': 1, 'cmap':
                 mpl.cm.coolwarm, 'linewidth': 0.01, 'antialiased': True,
                 'shade': True, 'alpha': .35}
    size = 50
    l = np.linspace(0.001, 0.999, size)
    x1_mesh, x2_mesh = np.meshgrid(l, l)
    x1_flat, x2_flat = x1_mesh.flatten(), x2_mesh.flatten()
    X_test = np.column_stack((x1_flat, x2_flat))
    y_test = franke_function(x1_flat, x2_flat)

    # NN setup
    NN = NeuralNetwork(
        layers=[2, 100, 60, 1], cost=MSE(),
        act_fns=["tanh", "tanh", "linear"]
    )
    NN.SGD(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=epochs, batch_size=100, eta=1e-2, reg=1e-2
    )

    # First frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("y")
    ax.set_zlim(-0.5, 1.5)
    # predict a mesh of data
    y_pred = np.load("./data/frames/frame0.npz")["y"]
    y_pred_mesh = np.reshape(y_pred, x1_mesh.shape)
    func = franke_function(x1_flat, x2_flat).reshape(size, size)
    plot = ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, **plot_args)
    plot2 = ax.plot_surface(x1_mesh, x2_mesh, func, alpha=.75)
    fig.colorbar(plot, shrink=0.5)

    def update_surf(num, x1_mesh, x2_mesh):
        y_pred = np.load("./data/frames/frame" + str(num) + ".npz")["y"]
        y_pred_mesh = np.reshape(y_pred, x1_mesh.shape)
        ax.clear()
        ax.set_zlim(-0.5, 1.5)
        plot = ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, **plot_args)
        plot2 = ax.plot_surface(x1_mesh, x2_mesh, func, alpha=.75)
        return plot, plot2

    ani = animation.FuncAnimation(
        fig, update_surf, epochs, fargs=(x1_mesh, x2_mesh), interval=333, blit=False
    )
    # plt.show()

    ani.save("./figures/franke.gif",  fps=3,  writer='imagemagick')
