import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from tqdm import tqdm

# Global parameters
np.random.seed(7)
DATA_FOLDER = Path("data/")
FIG_FOLDER = Path("fig/")


def preprocess_data(verbose=True, upsample=True):
    """
    Read the data file of red wine quality, a data set Randomizedwith 1599 data points,
    11 features and one categorical outcome (wine quality scored 0-10). The
    data set can be viewed as both a classification problem and a regression
    problem. Data set info:

    Predictors
    ----------
    fixed acidity
    volatile acidity
    citric acid
    residual sugar
    chlorides
    free sulfur dioxide
    total sulfur dioxide
    density
    pH
    sulphates
    alcohol

    Response
    --------
    quality : score 0-10

    Parameters
    ----------
    verbose : boolean
        Option to print a data set summary after preprocessing the data

    upsample : boolean
        Option to upsample training data to have equal amounts of Data
        for each category.

    Returns
    -------
    X, Xt : arrays, shape(N, p)
        Train and test data respectively,
        with shapes N data posints with p predictors

    y, yt : arrays, shape(N, 11)
        One-hot encoded response data with scores from 0-10 (11 categories)

    predictors : list
        List of all 12 feature names (11 predictors + 1 response)
    """
    df = pd.read_csv(DATA_FOLDER / "winequality-red.csv", sep=";")
    goodWines = df.loc[df["quality"] == 8]
    badWines = df.loc[df["quality"] == 3]
    if verbose:
        goodVolatileAcidity = goodWines["volatile acidity"]
        badVolatileAcidity = badWines["volatile acidity"]
        goodAlcohol = goodWines["alcohol"]
        badAlcohol = badWines["alcohol"]
        goodSulphates = goodWines["sulphates"]
        badSulphates = badWines["sulphates"]
        print(f"\n\t---------- Table 0: three most import predictors ----------")
        print(f"\t Quality | Volatile Acidity |     Alcohol    |  Sulphates  ")
        print(f"\t       8 |", end=" ")
        print(f"{np.mean(goodVolatileAcidity):1.2f} +/- {np.std(goodVolatileAcidity):1.2f}    |", end="")
        print(f"\t{np.mean(goodAlcohol):1.1f} +/- {np.std(goodAlcohol):1.1f} |", end="")
        print(f"\t{np.mean(goodSulphates):1.1f} +/- {np.std(goodSulphates):1.1f} ")
        print(f"\t       3 |", end=" ")
        print(f"{np.mean(badVolatileAcidity):1.2f} +/- {np.std(badVolatileAcidity):1.2f}    |", end="")
        print(f"\t{np.mean(badAlcohol):1.1f} +/- {np.std(badAlcohol):1.1f} |", end="")
        print(f"\t{np.mean(badSulphates):1.1f} +/- {np.std(badSulphates):1.1f} ", end="\n\n")

    X = df.loc[:, df.columns != "quality"].values
    y = df["quality"].values
    original_y = y
    predictors = list(df)  # list of all columns including response
    unitlist = [
        "[g(tartaric acit)/l]",
        "[g(acetic acit)/l]",
        "[g/l]",
        "[g/l]",
        "[g(sodium chloride)/l]",
        "[mg/l]",
        "[mg/l]",
        "[g/l]",
        "",
        "[g(potassium sulphate)/l]",
        "[vol.\%]",
    ]
    print("\n\t------------------ TABLE 1: raw X data summary ------------------")
    print("\tPredictor & Min & Max & Mean \\\\\\hline")
    for name, unit in zip(predictors[:-1], unitlist):
        # Table of some data properties
        min = np.min(df[name])
        max = np.max(df[name])
        mean = np.mean(df[name])
        s = name + " " + unit
        print(f"\t{s:35.35} & {min:6.3g} & {max:6.3g} & {mean:6.3g} \\\\")
    # Shortening names for later
    for i, name in enumerate(predictors[:-1]):
        if len(name) > 11:
            name = name[:9] + ".."
            predictors[i] = name
    # Split of train and test set
    X, Xt, y, yt = train_test_split(X, y, test_size=0.33)
    # Upscale so that the number of outputs in each class are equal
    if upsample:
        upscale = RandomOverSampler()
        X, y = upscale.fit_resample(X, y.astype("int"))
    # Scale both the train/test according to training set:
    scl = StandardScaler()
    scl.fit(X)
    X = scl.transform(X)
    Xt = scl.transform(Xt)
    # One hot encode the categorical outcome data
    # NOTE: It turns out that the data set only contains the categories
    # [3, 4, 5, 6, 7, 8] which are only 6 categories, instead of 11.
    enc = OneHotEncoder(sparse=False)
    y_enc = enc.fit_transform(y.reshape(-1, 1))
    yt_enc = enc.fit_transform(yt.reshape(-1, 1))
    score_categories = enc.categories_[0]
    num_categories = len(score_categories)
    # print a pretty overview of the data
    if verbose:
        # Info about X and Xt
        print("\n\t------------- TABLE 2: processed X data summary. ----------")
        print("\tPredictor   | train mean | train std | test mean | test std")
        print("\t------------|------------|-----------|-----------|---------")
        for i, name in enumerate(
            predictors[:-1]
        ):  # exclude response (from list of names)
            print(f"\t{name:11.11} ", end="")
            print(f"| {np.mean(X[:,i]):10.2e} | {np.std(X[:,i]):9.2f} ", end="")
            print(f"| {np.mean(Xt[:,i]):9.2e} | {np.std(Xt[:,i]):8.2f}")
        # Info about y and yt
        print("\n\t-- TABLE 3: processed y data summary ---")
        print("\tScore | train occurence | test occurence")
        print("\t------|-----------------|---------------")
        for i in range(num_categories):
            score = score_categories[i]
            ysum = np.sum(y_enc[:, i])
            ytsum = np.sum(yt_enc[:, i])
            print(f"\t{score:1.1f}   | {int(ysum):15} | {int(ytsum):14}")
        print("\n")
    return X, Xt, y, yt, predictors, score_categories, original_y


def plot_histogram(y, scores):
    # sns.distplot(y, kde=False)
    plt.bar(
        scores,
        height=[np.sum(y == i) for i in scores],
        width=1.0,
        color="lightblue",
        edgecolor="k",
        linewidth=1.0,
    )
    plt.grid()
    plt.ylabel("Occurence in Data Set")
    plt.xlabel("Sensory Score")
    plt.savefig(FIG_FOLDER / "histogram.png")
    plt.close()
    return None


def correlation_matrix(X, y, predictors):
    sigma = np.corrcoef(np.concatenate((X, y.reshape(-1, 1)), axis=1).T)
    ax = plt.subplot(111)
    ax.set_title(f"Correlation Matrix")
    map = sns.heatmap(
        sigma,
        ax=ax,
        vmin=-1.0,
        vmax=1.0,
        cmap="BrBG",
        cbar_kws={"ticks": np.linspace(-1, 1, 5)},
        annot=True,
        fmt="1.1f",
        xticklabels=predictors,
        yticklabels=predictors,
    )
    b, t = plt.ylim()  # discover the values for bottom and top
    plt.ylim(b + 0.5, t - 0.5)
    ax.set_xticklabels(predictors, rotation=45, ha="right")
    plt.tight_layout(pad=0.4, h_pad=1.0)
    plt.savefig(FIG_FOLDER / "corr_matrix.png")
    plt.close()
    return None


def accuracy(clf, X, y, cv=5):
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    acc = np.mean(scores)
    return acc


def score_wise_accuracy(yt, yp):
    print("\tScore |  3   |  4   |  5   |  6   |  7   |  8   |\n\t", end="T=0.5 | ")
    for i in range(3, 9):
        acc = np.sum(np.logical_and(yp == yt, yt == i)) / np.sum(yt == i)
        print(f"{acc:1.2f}", end=" | ")
    print("\n\tT=1.0 | ", end="")
    for i in range(3, 9):
        acc = np.sum(np.logical_and(np.abs(yt - yp) <= 1, yt == i)) / np.sum(yt == i)
        print(f"{acc:1.2f}", end=" | ")
    print("\n\n")
    return acc


def plot_confusion_matrix(X, Xt, y, yt, clf):
    clf.fit(X, y)
    yp = clf.predict(Xt)
    # Prints
    methodstr = type(clf.estimator).__name__
    f1 = f1_score(yt, yp, average="weighted")
    print("\t" + methodstr)
    print("\t" + "-" * len(methodstr))
    print(f"\tf1_score = {f1:1.2f}")
    print(f"\tAccuracy = {(100 * np.sum(yp == yt) / len(yt)):1.2f} %")
    for key in clf.best_params_:
        val = clf.best_params_[key]
        print(f"\t{key} = {val:,}")
    score_wise_accuracy(yt, yp)
    # Plot
    ax = plt.subplot(111)
    ax.set_title(methodstr)
    sns.heatmap(
        confusion_matrix(yt, yp),
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=",",
        xticklabels=range(3, 9),
        yticklabels=range(3, 9),
        square=True,
    )
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)
    plt.ylabel("True Score")
    plt.xlabel("Predicted Score")
    fn = methodstr + "_confusion_matrix_upsampled.png"
    plt.savefig(FIG_FOLDER / fn)
    plt.close()
    return None


def plot_feature_importances(X, y, clf, predictors):
    clf.fit(X, y)
    df = pd.DataFrame((clf.feature_importances_).reshape(1, -1), columns=predictors)
    df_sorted = df.sort_values(by=0, axis=1)
    # Plot
    methodstr = type(clf).__name__
    ax = plt.subplot(111)
    ax.set_title(methodstr)
    sns.barplot(data=df_sorted)
    plt.ylabel("Relative Importance")
    ax.set_xticklabels(df_sorted.columns, rotation=45, ha="right")
    ax.set_ylim(top=0.28)
    plt.grid()
    plt.tight_layout(pad=0.4, h_pad=1.0)
    fn = methodstr + "_feature_importance.png"
    plt.savefig(FIG_FOLDER / fn)
    plt.close()
    return None


def bagging(X, Xt, y, yt):
    print("Bagging\n-------")
    clf = GridSearchCV(
        BaggingClassifier(n_estimators=500, bootstrap=True, oob_score=True),
        param_grid={"n_estimators": range(100, 1600, 100)},
        cv=5,
        verbose=3,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )
    plot_confusion_matrix(X, Xt, y, yt, clf)
    # Terminal prints from two runs
    """
    [Parallel(n_jobs=-1)]: Done  75 out of  75 | elapsed:   51.1s finished
        BaggingClassifier
        -----------------
        f1_score = 0.66
        Accuracy = 67.61 %
        n_estimators = 1,000
        Score |  3   |  4   |  5   |  6   |  7   |  8   |
        T=0.5 | 0.00 | 0.04 | 0.79 | 0.69 | 0.55 | 0.20 |
        T=1.0 | 0.00 | 0.74 | 0.98 | 1.00 | 0.97 | 0.60 |

    UPSAMPLED
    [Parallel(n_jobs=-1)]: Done  75 out of  75 | elapsed:  1.2min finished
        BaggingClassifier
        -----------------
        f1_score = 0.64
        Accuracy = 64.77 %
        n_estimators = 800
        Score |  3   |  4   |  5   |  6   |  7   |  8   |
        T=0.5 | 0.00 | 0.05 | 0.79 | 0.62 | 0.53 | 0.12 |
        T=1.0 | 1.00 | 0.76 | 0.99 | 0.97 | 0.95 | 0.50 |
    """


def random_forest(X, Xt, y, yt):
    print("Random Forest\n-------------")
    clf = GridSearchCV(
        RandomForestClassifier(),
        param_grid={
            "n_estimators": range(100, 1100, 100),
            "max_features": range(1, 11, 1),
        },
        cv=5,
        verbose=3,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )
    plot_confusion_matrix(X, Xt, y, yt, clf)
    # Terminal prints from two runs
    """
    [Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:  2.4min finished
        RandomForestClassifier
        ----------------------
        f1_score = 0.65
        Accuracy = 66.48 %
        max_features = 9
        n_estimators = 100
        Score |  3   |  4   |  5   |  6   |  7   |  8   |
        T=0.5 | 0.00 | 0.00 | 0.78 | 0.68 | 0.53 | 0.20 |
        T=1.0 | 0.00 | 0.70 | 0.98 | 1.00 | 0.97 | 0.60 |

    UPSAMPLED
    [Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:  3.2min finished
        RandomForestClassifier
        ----------------------
        f1_score = 0.65
        Accuracy = 66.86 %
        max_features = 1
        n_estimators = 100
        Score |  3   |  4   |  5   |  6   |  7   |  8   |
        T=0.5 | 0.00 | 0.00 | 0.80 | 0.69 | 0.49 | 0.12 |
        T=1.0 | 0.33 | 0.76 | 1.00 | 0.99 | 0.99 | 0.62 |
    """


def boosting(X, Xt, y, yt):
    print("Gradient Boosting\n-----------------")
    clf = GridSearchCV(
        GradientBoostingClassifier(max_leaf_nodes=6),
        param_grid={
            "learning_rate": [0.05, 0.1, 0.15, 0.2],
            "n_estimators": range(100, 2100, 100),
        },
        cv=5,
        verbose=3,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )
    plot_confusion_matrix(X, Xt, y, yt, clf)
    # Terminal prints from two runs
    """
    [Parallel(n_jobs=-1)]: Done 380 out of 380 | elapsed:  5.4min finished
    GradientBoostingClassifier
        --------------------------
        f1_score = 0.64
        Accuracy = 64.96 %
        learning_rate = 0.15
        n_estimators = 200
        Score |  3   |  4   |  5   |  6   |  7   |  8   |
        T=0.5 | 0.00 | 0.13 | 0.73 | 0.68 | 0.50 | 0.20 |
        T=1.0 | 0.00 | 0.61 | 0.96 | 0.97 | 0.97 | 0.60 |

    UPSAMPLED
    [Parallel(n_jobs=-1)]: Done 400 out of 400 | elapsed: 10.8min finished
        GradientBoostingClassifier
        --------------------------
        f1_score = 0.64
        Accuracy = 64.39 %
        learning_rate = 0.2
        n_estimators = 200
        Score |  3   |  4   |  5   |  6   |  7   |  8   |
        T=0.5 | 0.00 | 0.00 | 0.72 | 0.68 | 0.57 | 0.12 |
        T=1.0 | 0.33 | 0.76 | 0.99 | 0.97 | 0.93 | 0.75 |
    """
