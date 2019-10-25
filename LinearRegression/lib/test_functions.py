import numpy as np
from .functions import *


def test_numpy_operator():
    """
    The functions uses numpy's @ operator, this test ensures that @ works
    as intended!
    """
    A = np.array([1,2,-1,3,0,4]).reshape(3,2)
    B = np.array([-1,-3,2,0]).reshape(2,2)
    numerical = A @ B
    # hand-calculated result:
    analytical= np.array([3,-3,7,3,8,0]).reshape(3,2)
    msg = "Error, there is something wrong with numpy's \"@\" operator"
    assert np.array_equal(numerical, analytical), msg


def test_polynomial_design_matrix():
    """
    Test that create_polynomial_design_matrix() works as intended
    """
    x = 2
    y = -1
    degree = 3
    A = np.array([x, y], dtype=np.float64).reshape((1,2))
    numerical = create_polynomial_design_matrix(A, degree)
    analytical = np.array([1,
        x, y,
        x**2, x*y, y**2,
        x**3, x**2*y, x*y**2, y**3
    ], dtype=np.float64).reshape(1,10)
    msg = "Error, there is something wrong with " + \
          "create_polynomial_design_matrix() in functions.py"
    assert np.array_equal(numerical, analytical), msg


def main():
    test_numpy_operator()
    test_polynomial_design_matrix()


main()
