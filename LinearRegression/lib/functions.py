from .regression import Regression
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from imageio import imread
import os

FIGURE_DIR = "./figures"
DATA_DIR = "./data"

if not os.path.exists(FIGURE_DIR):
    os.mkdir(FIGURE_DIR)

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

def image_path(fig_id):
    return os.path.join(FIGURE_DIR, fig_id)

def data_path(DATA_file):
    return os.path.join(DATA_DIR, DATA_file)

def save_fig(fig_id):
    fn = image_path(fig_id) + ".png"
    if os.path.exists(fn):
        overwrite = str(input("The file " + fn + " already exists," +
            "\ndo you wish to overwrite it [y/n/new_file_name]?\n"))
        if overwrite == "y":
            plt.savefig(fn, format="png")
            plt.close()
            print(fn + " was overwritten.")
        elif overwrite == "n":
            print("Figure was not saved.")
        elif fn != overwrite:
            fn = image_path(overwrite) + ".png" # user specified filename
            plt.savefig(fn, format="png")
            plt.close()
            print("New file: " + fn + " written.")
        # if user types new_file_name = fn, the original file is preserved,
        # and NOT overwritten.
    else:
        plt.savefig(fn, format="png")
        plt.close()
    return None


def create_polynomial_design_matrix(X_data, degree):
    """
    X_data = [x_data  y_data]
    Create polynomial design matrix on the form where columns are:
    X = [1  x  y  x**2  xy  y**2  x**3  x**2y  ... ]
    """
    X = PolynomialFeatures(degree).fit_transform(X_data)
    return X


def get_polynomial_coefficients(degree=5):
    """
    Return a list with coefficient names,
    [1  x  y  x^2  xy  y^2  x^3 ...]
    """
    names = ["1"]
    for exp in range(1,degree+1): # 0, ..., degree
        for x_exp in range(exp,-1,-1):
            y_exp = exp - x_exp
            if x_exp == 0:
                x_str = ""
            elif x_exp == 1:
                x_str = r"$x$"
            else:
                x_str = rf"$x^{x_exp}$"
            if y_exp == 0:
                y_str = ""
            elif y_exp == 1:
                y_str = r"$y$"
            else:
                y_str = rf"$y^{y_exp}$"
            names.append(x_str + y_str)
    return names


def franke_function(x, y):
    """
    Info:
    The Franke function f(x, y). The inputs are elements or vectors with
    elements in the domain of [0, 1].

    Inputs:
    x, y: array, int or float. Must be of same shape.

    Output:
    f(x,y), same type and shape as inputs.
    """
    if np.shape(x) != np.shape(y):
        raise ValueError("x and y must be of same shape!")

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Ordinary least squares analysis

def ols_franke_function(X, y, x1, x2, variance, degree=5, svd=False, k=5):
    """
    Info:
    Make a plot of the sample data and an Ordinary Least Squares fit on
    the data. The purpose here is only to illustrate the fit, and the
    data is NOT yet split in training/test data.

    Input:
    * X: design matrix
    * y: sample data
    * x1: sample data
    * x2: sample data
    * degree=5: polynomial degree of regression.
    * svd=False: if set to true the matrix inversion of (X.T @ X) will
        be inverted with SVD
    * k=5: number of k subsets for k-fold CV

    Output:
    The function produces a plot of the data and corresponding regression.
    """
    # OLS
    model = Regression(X, y)
    model.ols_fit(svd=svd)
    y_pred = model.predict(X)
    N = X.shape[0]
    mse = model.mean_squared_error(y, y_pred)
    r2 = model.r2_score(y, y_pred)

    # PLOT
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # plot the original data
    data = ax.scatter(x1, x2, y, marker="^", alpha=0.04)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("y")
    # make a surface plot on a smooth meshgrid, given OLS model
    l = np.linspace(0,1,1001)
    x1_mesh, x2_mesh = np.meshgrid(l, l)
    x1_flat, x2_flat = x1_mesh.flatten(), x2_mesh.flatten()
    X_mesh = np.column_stack((x1_flat, x2_flat))
    y_pred = model.predict(create_polynomial_design_matrix(X_mesh, degree))
    y_pred_mesh = np.reshape(y_pred, x1_mesh.shape)
    surface = ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh,
        cmap=mpl.cm.coolwarm)
    fake_surf = mpl.lines.Line2D([0],[0], linestyle="none", c="r",
        marker="s", alpha=.5)
    fake_data = mpl.lines.Line2D([0],[0], linestyle="none", c="b",
        marker="^", alpha=.2)
    ax.legend([fake_surf, fake_data], \
        [f"OLS surface fit with polynomial degree {degree}\n" + \
        "Training MSE = " + f"{mse:1.3f}" + r", $R^2$ = " + f"{r2:1.3f}", \
        f"{N} Data points with variance = {variance:1.3f}"],numpoints=1, loc=1)
    fig.colorbar(surface, shrink=0.5)
    save_fig("ols_franke_function")
    return None


def ols_test_size_analysis(X, y, variance, svd=False):
    """
    Info:
    Analyse the MSE and R2 as a function of test size

    Input:
    * X: design matrix
    * y: y data
    * variance: Var(y) in noise of franke function, only used for plot title
    * svd=False: if set to true the matrix inversion of (X.T @ X) will
        be inverted with SVD

    Output:
    Produces and saves a plot
    """
    N = X.shape[0]
    n = 17
    test_sizes = np.linspace(0.1,0.9,n)
    mse = np.zeros(n)
    r2 = np.zeros(n)
    model = Regression(X, y)
    # Collect MSE and R2 as a function of test_size
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=test_sizes[i], random_state=i)
        model.update_X(X_train)
        model.update_y(y_train)
        model.ols_fit(svd=svd)
        y_pred = model.predict(X_test)
        mse[i] = model.mean_squared_error(y_test, y_pred)
        r2[i] = model.r2_score(y_test, y_pred)
    # Plot the results
    plt.subplot(211)
    plt.title(f"Test Size Analysis, Data points = {N}, Variance = {variance:1.3f}")
    plt.plot(test_sizes, mse, label="Mean Squared Error")
    plt.ylabel("MSE"); plt.grid(); plt.legend()
    plt.subplot(212)
    plt.plot(test_sizes, r2, label="R squared score")
    plt.ylabel(r"$R^2$"); plt.grid(); plt.legend()
    plt.xlabel("Test Size")
    save_fig("ols_test_size_analysis")
    return None


def ols_beta_variance(X_data, y, variance=1., degree=5):
    """
    Info:
    plot the beta-coefficients with errorbars corresponding to one sigma
    (standard deviation) = sqrt(variance)

    Input:
    * X_data: x1, x2 coordinates of data
    * y: y coordinates of data
    * variance=1: sigma**2 of noise in data
    * degree=5: polynomial degree for design matrix

    Output:
    Produces and saves a plot
    """
    p = X.shape[1]
    XTXinv = np.linalg.inv(X.T @ X)
    x = np.linspace(0,p-1,p)
    beta = XTXinv @ X.T @ y
    beta_err = np.diag(XTXinv)*np.sqrt(variance)
    names = get_polynomial_coefficients(degree)
    # PLOT
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.errorbar(x, beta, yerr=beta_err, fmt="b.", capsize=3,
        label=r"$\beta_i\pm\sigma$")
    ax.set_title(r"$\beta_i$ confidence intervals")
    ax.set_xticks(x.tolist())
    ax.set_xticklabels(names, fontdict={"fontsize": 7})
    plt.ylabel(r"$\beta$")
    plt.xlabel(r"$\beta$ coeff. terms")
    plt.grid()
    plt.legend()
    save_fig("beta_st_dev")
    return None


def ols_k_fold_analysis(X, y, variance, largest_k, svd=False):
    """
    Info:
    Analyse the MSE and R2 as a function of k

    Input:
    * X: design matrix
    * y: y data
    * variance: Var(y) in noise of franke function, only used for plot title
    * largest_k: integer where k = [2, 3, ..., largest_k]
    * svd=False: if set to true the matrix inversion of (X.T @ X) will
        be inverted with SVD

    Output:
    Produces and saves a plot
    """
    N = X.shape[0]
    if largest_k < 2:
        raise ValueError("largest k must be >= 2.")
    n = largest_k - 1
    k_arr = np.linspace(2,largest_k,n, dtype=np.int64)
    mse = np.zeros(n)
    r2 = np.zeros(n)
    model = Regression(X, y)
    # Collect MSE and R2 as a function of test_size
    for i in range(n):
        mse[i], r2[i] = model.k_fold_cross_validation(k_arr[i], "ols", svd=svd)
    # Plot the results
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    plt.title(f"K-fold Cross Validation, Data points = {N}, Variance = {variance:1.3f}")
    plt.plot(k_arr, mse, label="Mean Squared Error")
    plt.ylabel("MSE"); plt.grid(); plt.legend()
    ax1.set_yticklabels([f"{x:1.4f}" for x in ax1.get_yticks().tolist()])
    ax2 = fig.add_subplot(2,1,2)
    plt.plot(k_arr, r2, label="R squared score")
    plt.ylabel(r"$R^2$"); plt.grid(); plt.legend()
    plt.xlabel("k")
    ax2.set_yticklabels([f"{x:1.4f}" for x in ax2.get_yticks().tolist()])
    save_fig("ols_k_fold_analysis")
    return None


def ols_degree_analysis(X_data, y, min_deg, max_deg, variance, svd=False, k=5):
    """
    Info:
    Analyse the MSE as a function of degree

    Input:
    * X: design matrix
    * y: y data
    * min_deg: maximum polynomial degree
    * max_deg: minimum polynomial degree
    * variance: Var(y) in noise of franke function, only used for plot title
    * svd=False: if set to true the matrix inversion of (X.T @ X) will
        be inverted with SVD
    * k=5: number of k subsets for k-fold CV

    Output:
    Produces and saves a plot
    """
    N = X_data.shape[0]
    n = max_deg - min_deg + 1
    mse_k_fold = np.zeros(n)
    mse_train = np.zeros(n)
    degree_arr = np.linspace(min_deg,max_deg,n, dtype=np.int64)
    model = Regression(X_data, y)
    min_MSE = variance*5
    min_deg = 0
    # Collect MSE and R2 as a function of test_size
    for i in range(n):
        deg = degree_arr[i]
        X = create_polynomial_design_matrix(X_data, deg)
        model.update_X(X)
        model.ols_fit(svd=svd)
        y_pred = model.predict(X)
        mse_train[i] = model.mean_squared_error(y, y_pred)
        mse_k_fold[i], r2 = model.k_fold_cross_validation(k, "ols", svd=svd)
        if mse_k_fold[i] < min_MSE:
            min_deg = deg
            min_MSE = mse_k_fold[i]
    # Plot the results
    fig = plt.figure()
    plt.title(f"Data points = {N}, Variance = {variance:1.4f}")
    plt.plot(degree_arr, mse_train, label=f"MSE: training data")
    plt.plot(degree_arr, mse_k_fold, label=f"MSE: {k}-fold CV")
    plt.plot(min_deg, min_MSE, "ro", label=f"Min. MSE = {min_MSE:1.4f}")
    plt.plot()
    plt.ylabel("Prediction Error")
    plt.xlabel("Polynomial degree")
    plt.grid()
    plt.legend()
    save_fig("degree_analysis_ols")
    return None


def ols_degree_and_n_analysis(max_log_N, max_degree, test_size=0.33, st_dev=.5,
    svd=False, k=5):
    """
    Info:
    Study the prediction error vs. degree and number of data points, and also
    the prediction error will be evaluated on BOTH the training set and
    the test set. This should illustrate the Bias-Variance tradeoff (optimal
    degree of complexity) and how the training set gets overfitted.

    Input:
    * max_log_N: Largest exponent of 10 (int > 3). The number of data points
                 will be N = 10**(log N), where
                 - log N = [3, 4, ...]
                 - N = [100, 1000, 10 000, ...]
    * max_degree: Largest polynomial degree for OLS regression.
                  Integer between 0 and 20
    * test_size: fraction of data which will be used in test set
    * st_dev: standard deviation on noise in franke function
    * svd=False: if set to true the matrix inversion of (X.T @ X) will
        be inverted with SVD
    * k=5: number of k subsets for k-fold CV

    Output:
    Produces and saves a plot
    """
    if max_log_N <= 3:
        raise ValueError("max_log_N must be an integer > 3")
    if max_degree < 0 or max_degree > 20:
        raise ValueError("max_degree should be an integer between 0 and 20")

    degree_arr = np.linspace(2,max_degree,max_degree-1, dtype=np.int64)
    N_arr = np.logspace(3,max_log_N,(max_log_N-2), dtype=np.int64)
    log_N_arr = np.linspace(3,max_log_N,(max_log_N-2), dtype=np.int64)
    log_N_mesh, degree_mesh = np.meshgrid(log_N_arr, degree_arr)
    mse_test_mesh = np.zeros(log_N_mesh.shape)
    mse_train_mesh = np.zeros(log_N_mesh.shape)
    least_MSE_for_this_N = np.ones(len(log_N_arr))*100
    least_MSE_degree = np.zeros(len(log_N_arr))
    min_MSE = 100

    # loop over N
    for i,N in enumerate(N_arr):
        # Franke function data with N data points
        x1 = np.random.uniform(0, 1, N)
        x2 = np.random.uniform(0, 1, N)
        X_data = np.column_stack((x1, x2))
        y = franke_function(x1, x2) + np.random.normal(0, st_dev, N)
        # loop over complexity (degree)
        for j,deg in enumerate(degree_arr):
            X = create_polynomial_design_matrix(X_data, deg)
            model = Regression(X, y)
            model.ols_fit(svd=svd)
            y_pred = model.predict(X) # predict on training set
            # Evaluate MSE with 5 fold CV
            mse, r2 = model.k_fold_cross_validation(k, "ols", svd=svd)
            mse_test_mesh[j,i] = mse
            mse_train_mesh[j,i] = model.mean_squared_error(y, y_pred)
            if mse < least_MSE_for_this_N[i]:
                least_MSE_for_this_N[i] = mse
                least_MSE_degree[i] = deg

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"OLS on Franke's function with \n" + \
        "various number of data points.\n\n")
    test_surf = ax.plot_surface(log_N_mesh, degree_mesh, mse_test_mesh,
        color="yellow",alpha=.5)
    train_surf = ax.plot_surface(log_N_mesh, degree_mesh, mse_train_mesh,
        color="b",alpha=.5)
    least_mse = ax.scatter(log_N_arr, least_MSE_degree, least_MSE_for_this_N,
        c="r", marker="o")
    ax.set_xticks(log_N_arr.tolist())
    ax.set_yticks(degree_arr.tolist())
    ax.set_yticklabels([f"{int(y):,}" for y in ax.get_yticks().tolist()]) # Deg
    ax.set_xticklabels([f"{int(x):,}" for x in N_arr.tolist()]) # N
    ax.set_ylabel("Polynomial Degree")
    ax.set_xlabel(r"$N$")
    ax.set_zlabel("Prediction Error")
    # Legend: because of matplotlib, we create two fake lines in the same
    # color as the surface plots, and use these for the legend
    fake_test = mpl.lines.Line2D([0],[0], linestyle="none", c="yellow",
        marker="s", alpha=.8)
    fake_train = mpl.lines.Line2D([0],[0], linestyle="none", c="b",
        marker="s", alpha=.5)
    ax.legend([fake_test, fake_train, least_mse],
        [f"MSE: {k}-fold CV", "MSE: Training data", \
        "Tradeoff (minimum MSE on 5-fold CV)"], \
        numpoints=1, loc=1)
    plt.show()
    save_fig("ols_degree_and_n_analysis")
    return None

# Ridge/Lasso analysis

def lambda_and_degree_analysis(X_data, y, degree_arr, lambda_arr, method,
    variance, k=5):
    """
    Info:
    Study the prediction error vs. degree and lambda for Ridge/Lasso regression.
    This should illustrate the optimal degree/lambda in pursuit of the best
    model.

    Input:
    * X_data: matrix where columns are the (x1, x2) coordinates of the data
    * y: array with the y coordinates of the data
    * degree_arr: array of polynomial degrees (dtype=np.int64)
    * lambda_arr: array of log_(10) of lambda values (dtype=np.int64)
    * method: "ridge" of "lasso"
    * variance: Var(y) in noise of franke function, only used for plot title
    * k=5: number of k subsets for k-fold CV

    Output:
    Produces and saves a plot
    """
    degree_mesh, lambda_mesh = np.meshgrid(degree_arr, lambda_arr)
    mse_mesh = np.zeros(degree_mesh.shape)
    least_mse = 10000 # arbitrary high number
    least_deg = 0
    least_lam = 0
    model = Regression(X_data, y)

    # loop over complexity (degree)
    for i in range(len(degree_arr)):
        deg = degree_arr[i]
        X = create_polynomial_design_matrix(X_data, deg)
        model.update_X(X)
        # loop over lambdas
        for j in range(len(lambda_arr)):
            lam = 10.0**lambda_arr[j]
            mse, r2 = model.k_fold_cross_validation(k, method, alpha=lam)
            mse_mesh[j,i] = mse
            if mse < least_mse:
                least_mse = mse
                least_deg = deg
                least_lam = lambda_arr[j]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if method == "ridge":
        m = "Ridge"
    else:
        m = "Lasso"
    ax.set_title(m + f" regression: MSE, evaluated with {k}-fold CV,\n" + \
        f"{X.shape[0]} data points with variance = {variance:1.4f}",
        loc="left")
    minimum = ax.scatter(least_deg, least_lam, least_mse, c="r", marker="o")
    test_surf = ax.plot_surface(degree_mesh, lambda_mesh, mse_mesh,
        cmap=mpl.cm.coolwarm,alpha=.75,norm=mpl.colors.PowerNorm(gamma=0.5))
    fig.colorbar(test_surf, shrink=0.7, extend="both")
    ax.set_xticks(degree_arr[::2].tolist()) # every third degree shown
    ax.set_yticks(lambda_arr[::2].tolist()) # every second log(lam) shown
    ax.set_xticklabels([f"{int(x):,}" for x in ax.get_xticks().tolist()])
    ax.set_yticklabels([f"{int(y):,}" for y in ax.get_yticks().tolist()])
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel(r"$\log_{10}(\lambda)$")
    ax.set_zlabel("Prediction Error")
    # Legend
    ax.legend([minimum],[rf"Min. MSE = {least_mse:1.4f}, " + \
        rf"$\log(\lambda)$ = {least_lam}, degree = {least_deg}"], \
        numpoints=1, loc="upper right")
    save_fig("lambda_and_degree_analysis_" + method)
    return None

# Terrain analysis

# OLS
def OLS_regression_on_terrain(filename, degree_arr, k=5, svd=False):
    """
    Info:
    Perform OLS regression on ".tif" terrain data for all degrees
    in degree_arr. MSE is evaluated by using k-fold CV.

    Input:
    * filename: names of files in data folder. The files must be .tif format,
        but the ".tif" should not be included in the argument.
    * degree_arr: array of desired polynomial degrees to fit data with
    * k=5: number of k subsets for k-fold CV
    * svd: option for matrix inversion by SVD

    Output:
    Produces and saves a plot.
    """
    fn = data_path(filename + ".tif")
    nth = 3
    terrain = np.asarray(imread(fn))[::nth,::nth]*0.001 # height rel. to sea [km]
    ny, nx = terrain.shape
    # positive x-direction=EAST, positive y-direction=NORTH:
    terrain = np.flip(terrain, axis=0)
    # create corresponding x,y meshgrid:
    x_len_km = nth*0.031*(nx-1) # distance between points
    y_len_km = nth*0.031*(ny-1)
    x = np.linspace(0,x_len_km,nx)
    y = np.linspace(0,y_len_km,ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    shape = x_mesh.shape
    xi = x_mesh.flatten()
    yi = y_mesh.flatten()
    y_data = terrain.flatten()
    X_data = np.column_stack((xi,yi))
    num_deg = len(degree_arr)
    mse_arr = np.zeros(num_deg)
    mse_k_fold = np.zeros(k)
    # loop over different degrees
    for deg in range(num_deg-1,-1,-1):
        degree = int(degree_arr[deg])
        print(f"Degree {degree_arr[deg]}")
        X = create_polynomial_design_matrix(X_data, degree)
        model = Regression(X, y_data)
        mse, r2 = model.k_fold_cross_validation(k, "ols", svd=svd)
        mse_arr[deg] = mse # [km]
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(degree_arr, mse_arr)
    i = np.argmin(mse_arr)
    plt.plot(degree_arr[i], mse_arr[i], 'ro', label=f"Minimum = {mse_arr[i]:1.3f}")
    ax.set_title("OLS regression: MSE, evaluated {k}-fold CV" + \
        f" with {k} subsets", loc="left")
    ax.set_xticks(degree_arr.tolist())
    plt.ylabel("MSE [km]")
    plt.xlabel("Polynomial degree")
    plt.grid()
    plt.legend()
    save_fig("terrain_ols_degree_analysis")
    return None

# Ridge/Lasso
def regression_on_terrain_ridge_lasso(filename, method, degree_arr,
    lambda_arr, k=5):
    """
    Info:
    Perform RIDGE/LASSO regression on ".tif" terrain data for all degrees
    in degree_arr and all alphas in 10**lambda_arr.
    MSE is evaluated by using k-fold CV.

    Input:
    * filename: names of files in data folder. The files must be .tif format,
        but the ".tif" should not be included in the argument.
    * degree: polynomial degree to fit data with
    * method: ols, ridge or lasso
    * alpha: parameter for ridge/lasso

    Output:
    Produces and saves a plot.
    """
    fn = data_path(filename + ".tif")
    nth = 3
    terrain = np.asarray(imread(fn))[::nth,::nth]*0.001 # km, every third point
    ny, nx = terrain.shape
    # positive x-direction=EAST, positive y-direction=NORTH:
    terrain = np.flip(terrain, axis=0)
    # create corresponding x,y meshgrid:
    x_len_km = nth*0.031*(nx-1) # distance between points
    y_len_km = nth*0.031*(ny-1)
    x = np.linspace(0,x_len_km,nx)
    y = np.linspace(0,y_len_km,ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    shape = x_mesh.shape
    xi = x_mesh.flatten()
    yi = y_mesh.flatten()
    y_data = terrain.flatten()
    X_data = np.column_stack((xi,yi))
    degree_mesh, lambda_mesh = np.meshgrid(degree_arr, lambda_arr)
    mse_mesh = np.zeros(degree_mesh.shape)
    num_deg = len(degree_arr)
    num_lam = len(lambda_arr)
    least_mse = 1e8 # arbitrary large number
    # loop over degree
    for j,degree in enumerate(degree_arr):
        print(f"Degree {degree}")
        X = create_polynomial_design_matrix(X_data, degree)
        model = Regression(X, y_data)
        # loop over lambdas
        for i in range(num_lam):
            lam = 10.**lambda_arr[i]
            print(f"    alpha {lam:1.2g}")
            mse, r2 = model.k_fold_cross_validation(k, method, alpha=lam)
            mse_mesh[i,j] = mse
            if mse < least_mse:
                least_deg = degree
                least_lam = lambda_arr[i]
                least_mse = mse

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if method == "ridge":
        method_title = "Ridge "
    else:
        method_title = "Lasso "
    ax.set_title(method_title + "regression: MSE, evaluated with " + \
        f"{k}-fold CV", loc="left")
    minimum = ax.scatter(least_deg, least_lam, least_mse, c="r", marker="o")
    test_surf = ax.plot_surface(degree_mesh, lambda_mesh, mse_mesh,
        cmap=mpl.cm.coolwarm,alpha=.75,norm=mpl.colors.PowerNorm(gamma=0.5))
    fig.colorbar(test_surf, shrink=0.7, extend="both")
    ax.set_xticks(degree_arr[::].tolist())
    ax.set_yticks(lambda_arr[::].tolist())
    ax.set_xticklabels([f"{int(x):,}" for x in ax.get_xticks().tolist()])
    ax.set_yticklabels([f"{int(y):,}" for y in ax.get_yticks().tolist()])
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel(r"$\log_{10}(\lambda)$")
    ax.set_zlabel("Prediction Error")
    # Legend
    ax.legend([minimum],[rf"Min. MSE = {least_mse:1.4f}, " + \
        rf"$\log(\lambda)$ = {least_lam}, degree = {least_deg}"], \
        numpoints=1, loc="upper right")
    save_fig("terrain_" + method + "_lambda_and_degree_analysis")
    return None
