from lib.functions import *
import lib.test_functions  # all tests are executed by import
import numpy as np

np.random.seed(13)


def franke_analysis():
    """
    Comment/uncomment desired analysis functions
    """
    # Create Franke function data:
    N = 10000
    degree = 5
    st_dev = 0.5
    variance = st_dev ** 2
    x1 = np.random.uniform(0, 1, N)
    x2 = np.random.uniform(0, 1, N)
    X_data = np.column_stack((x1, x2))
    f = franke_function(x1, x2)
    # create y data based on franke function with noise
    noise = np.random.normal(0, st_dev, N)
    y = f + noise  # y_data
    variance = np.var(noise)
    # polynomial design matrix for x data
    X = create_polynomial_design_matrix(X_data, degree)

    # ORDINARY LEAST SQUARES
    # ols_franke_function(X, y, x1, x2, variance, degree, svd=False)

    # ols_beta_variance(X_data, y, variance=variance, degree=degree)
    # RESAMPLING TECHNIQUES
    # ols_test_size_analysis(X, y, variance, svd=True)
    # ols_k_fold_analysis(X, y, variance, largest_k=15, svd=True)
    # ols_degree_analysis(X_data, y, 2, 11, variance, svd=True)
    # ols_degree_and_n_analysis(max_log_N=6, max_degree=10, st_dev=st_dev)

    # LASSO/RIDGE LAMBDA AND DEGREE ANALYSIS
    min = 2
    max = 15
    degree_arr = np.linspace(min, max, max - min + 1, dtype=np.int32)
    lambda_arr = np.linspace(-11, 0, 12, dtype=np.int32)
    lambda_and_degree_analysis(X_data, y, degree_arr, lambda_arr, "ridge", variance)
    lambda_arr = np.linspace(-13, -3, 11, dtype=np.int32)
    # lambda_and_degree_analysis(X_data, y, degree_arr, lambda_arr, "lasso",
    # variance)
    return None


def terrain_analysis():
    degree_arr = np.arange(5, 11 + 1, dtype=np.int32)
    # OLS_regression_on_terrain("dead_sea", degree_arr, k=5)
    degree_arr = np.arange(5, 11 + 1, dtype=np.int32)
    lambda_arr = np.arange(-3, -5 - 1, -1, dtype=np.int32)
    # regression_on_terrain_ridge_lasso("dead_sea", "ridge", degree_arr,
    # lambda_arr, k=5)
    degree_arr = np.arange(5, 12 + 1, dtype=np.int32)
    lambda_arr = np.arange(-3, -6 - 1, -1, dtype=np.int32)
    regression_on_terrain_ridge_lasso("dead_sea", "lasso", degree_arr, lambda_arr, k=5)


def main():
    # franke_analysis()
    terrain_analysis()


main()
