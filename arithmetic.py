import functools
import numpy as np


def pr(a):
    """
    For vectors A, divide each element of A by the last.
    For matrices, divide each row by its rightmost element.
    For arrays of high dimension, project along the last dimension.
    """
    a = np.asarray(a)
    return a[..., :-1] / a[..., -1:]


def unpr(a):
    """
    For vectors A, return A with 1 appended.
    For matrices, append a column of ones.
    For arrays of higher dimension, append along the last dimension.
    """
    a = np.asarray(a)
    col_shape = a.shape[:-1] + (1,)
    return np.concatenate((a, np.ones(col_shape)), axis=-1)


def normalized(a):
    """
    For vertors A, return a unit vector in the direction of A.
    For matrices, return a matrix of the same size where each row has unit norm.
    For arrays of higher dimension, normalize along the last axis.
    """
    a = np.asarray(a)
    return a / np.sqrt(np.sum(np.square(a), axis=-1))[..., None]


def unreduce(a, mask, fill=0.):
    """
    Create a vector in which the positions set to True in the mask vector are set to sequential values from A.
    """
    a = np.asarray(a)
    mask = np.asarray(mask)
    x = np.repeat(fill, len(mask)).astype(a.dtype)
    x[mask] = a
    return x


def unreduce_2d(a, mask, fill=0.):
    """
    Create a matrix in which the rows and columns set to True in the mask vector are set to the entries from A.
    """
    a = np.asarray(a)
    x = np.ones((len(mask), len(mask))) * fill
    x[np.ix_(mask, mask)] = a
    return x


def sumsq(a, axis=None):
    """
    Compute the sum of squares.
    """
    return np.sum(np.square(a), axis=axis)


def skew(a):
    """
    Compute the skew-symmetric matrix for a. The returned matrix M has the property 
    that for any 3-vector v, M * v = a x v where x denotes the cross product.
    """
    a = np.asarray(a)
    assert a.shape == (3,), 'shape was was %s' % str(a.shape)
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0.]])


def unit(i, n):
    """
    Compute a unit vector along the i-th axis in N dimensions.
    """
    return (np.arange(n) == i).astype(float)


def orthonormalize(a):
    """
    Find the closest orthonormal matrix to R.
    Note that the returned matrix may have determinant +1 or -1, so the result
    may be either a rotation or a rotoinversion.
    """
    u, _, v = np.linalg.svd(a)
    return np.dot(u, v)


def minmedmax(a):
    """
    Compute [min(x), median(a), max(a)]
    """
    if len(a) == 0:
        raise Exception('warning [utils.minmedmax]: empty list passed')
    else:
        return np.min(a), np.median(a), np.max(a)


def cis(th):
    """
    Compute [cos(th), sin(th)].
    """
    return np.array((np.cos(th), np.sin(th)))


def dots(*a):
    """
    Multiply an arbitrary number of matrices with np.dot.
    """
    return functools.reduce(np.dot, a)
