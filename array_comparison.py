import numpy as np

from .arithmetic import spy


def assert_arrays_almost_equal(x, y, decimals=6, linelimit=20, xlabel='LHS', ylabel='RHS'):
    """
    Check that x and y are equal to within the specified number of decimal places.
    If they are not then print some helpful diagnostic information and raise an AssertionError.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, 'Shape mismatch: %s vs %s' % ('x'.join(map(str, x.shape)), 'x'.join(map(str, y.shape)))
    tol = np.maximum(10**-decimals, np.abs(np.maximum(x, y) * 10**-decimals))
    mismatches = np.abs(x - y) > tol
    if np.any(mismatches):
        error_coords = list(zip(*np.nonzero(mismatches)))
        for coord in error_coords[:linelimit]:
            print('Mismatch at position %s: %s=%s vs %s=%s' % (','.join(map(str, coord)), xlabel, x[coord], ylabel, y[coord]))
        if len(error_coords) > linelimit:
            print('... and %d more' % (len(error_coords) - linelimit))
        if x.ndim > 1 and np.prod(x.shape) > 10:
            print('\n' + xlabel + ':')
            print(spy(x))
            print('\n' + ylabel + ':')
            print(spy(y))
            print('\nError:')
            print(spy(mismatches))
        else:
            print('\n:' + xlabel + ':')
            print(x)
            print('\n' + ylabel + ':')
            print(y)
            print('\nSparsity pattern of difference:')
            print(spy(mismatches))
        raise AssertionError('arrays were not equal to %d decimals' % decimals)
