import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def read_file(filename):
    """Read a csv data file located in the ./data/ directory
    Return design matrix X and y-data"""
    DATA_DIR = "./data"
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    DATA_FILE = os.path.join(DATA_DIR,filename)
    df = pd.read_csv(DATA_FILE).values
    y = np.array(df[:,1])
    X = np.array(df[:,2:])
    return X, y


def save_fig(y_test, y_pred, score, fig_id):
    """Save a matplotlib figure to the ./figure/ directory.
    If/else statements ask the user if they wish to overwrite an
    already existing figure."""
    # Plot
    l = y_test.shape[0]
    x = np.linspace(1,l,l)
    plt.plot(y_test, y_pred, 'ro', label=fr"prediction, $R^2 = ${score:2.2f}")
    min = np.min(y_test)
    max = np.max(y_test)
    plt.plot([min,max],[min,max])
    plt.xlabel("CADi true values")
    plt.ylabel("CADi prediction")
    plt.legend()
    plt.grid()
    # Saving of figure
    FIGURE_DIR = "./figures"
    if not os.path.exists(FIGURE_DIR):
        os.mkdir(FIGURE_DIR)
    FIGURE_FILE = os.path.join(FIGURE_DIR, fig_id)
    if os.path.exists(FIGURE_FILE):
        overwrite = str(input("The file \"" + FIGURE_FILE + \
            "\" already exists, do you wish to overwrite it [y/n]?\n"))
        if overwrite == "y":
            plt.savefig(FIGURE_FILE, format="png")
            plt.close()
            print("Figure was overwritten.")
        else:
            print("Figure was not saved.")
    else:
        plt.savefig(FIGURE_FILE, format="png")
        plt.close()
    return None
