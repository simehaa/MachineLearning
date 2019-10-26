import numpy as np
from .functions import *


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


def main():
    test_numpy_operator()


main()
