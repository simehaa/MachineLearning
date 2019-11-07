import numpy as np
from lib.functions import *
from lib.linear_regression import *


def test_numpy_operator():
    """
    The functions uses numpy's @ operator, this test ensures that @ works
    as intended!
    """
    A = np.array([1, 2, -1, 3, 0, 4]).reshape(3, 2)
    B = np.array([-1, -3, 2, 0]).reshape(2, 2)
    numerical = A @ B
    # hand-calculated result:
    analytical = np.array([3, -3, 7, 3, 8, 0]).reshape(3, 2)
    msg = 'Error, there is something wrong with numpy\'s "@" operator'
    assert np.array_equal(numerical, analytical), msg


def test_ridge_regression():
    """
    Test a simple linear function f(x) = 4x + 2, and checks that Ridge
    regression obtains the coefficients 4 and 2 (+/- 0.1).
    """
    x = np.linspace(-2,2,1000)
    X = np.column_stack((np.ones(1000), x))
    y = 4*x + 2 + np.random.normal(0,0.2,1000)
    LinReg = LinearRegression(X, y)
    LinReg.ridge_fit()
    intercept, slope = LinReg.beta
    msg = "Error, a simple Ridge regression did not correctly fit the data:\n"
    msg += f"\t\tOriginal: f(x) = 4.0x + 2.0,\n"
    msg += f"\t\tObtained: g(x) = {slope:2.1f}x + {intercept:2.1f}."
    assert np.abs(intercept - 2) < 0.1 and np.abs(slope - 4) < 0.1, msg


def test_franke_noise():
    """
    Test that the noise and mean in generate_franke_data() actually
    Returns whats inteded.
    """
    X, y_noise = generate_franke_data(N=10000, noise=0.5)
    y_func = franke_function(X[:,0], X[:,1])
    noise = np.std(y_noise - y_func)
    mean = np.mean(y_noise - y_func)
    msg = "Error, generate franke data did not create noise as intended.\n"
    msg += f"\t\tExpected mean=0, std=0.5, Obtained mean={mean:1.1f}, std={noise:1.1f}."
    assert np.abs(noise - 0.5) < 0.1 and np.abs(mean) < 0.1, msg


def tests():
    test_numpy_operator()
    test_ridge_regression()
    test_franke_noise()
    print("All test functions complete.\n")
