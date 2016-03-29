import numpy as np
import scipy.linalg

from .arithmetic import skew

# Cached for efficiency...
_eye3 = np.eye(3)

# Generators of the 3x3 Lie group sl(3)
generators = np.array([
	[[1, 0, 0],
	 [0, -1, 0],
	 [0, 0, 0]],

	[[0, 0, 0],
	 [0, -1, 0],
	 [0, 0, 1]],

	[[0, 1, 0],
	 [0, 0, 0],
	 [0, 0, 0]],

	[[0, 0, 1],
	 [0, 0, 0],
	 [0, 0, 0]],

	[[0, 0, 0],
	 [0, 0, 1],
	 [0, 0, 0]],

	[[0, 0, 0],
	 [1, 0, 0],
	 [0, 0, 0]],

	[[0, 0, 0],
	 [0, 0, 0],
	 [1, 0, 0]],

	[[0, 0, 0],
	 [0, 0, 0],
	 [0, 1, 0]],
], dtype=float)


def hat(v):
	"""
	Given an 8-vector, compute a 3x3 matrix with trace zero.
	"""
	return np.array([
		[v[0],  v[2],      v[3]],
		[v[5], -v[0]-v[1], v[4]],
		[v[6],  v[7],      v[1]]
	])


def unhat(m):
	"""
	Given a 3x3 matrix M with trace zero, compute a vector v such that hat(v) = M
	"""
	return np.array([
		m[0, 0],
		m[2, 2],
		m[0, 1],
		m[0, 2],
		m[1, 2],
		m[1, 0],
		m[2, 0],
		m[2, 1],
	])


def multiply_generators(v):
	"""
	Given a 3-vector v, compute the product of each of the i generatosr with v.
	"""
	return np.array([
		[v[0], -v[1], 0],
		[0, -v[1], v[2]],
		[v[1], 0, 0],
		[v[2], 0, 0],
		[0, v[2], 0],
		[0, v[0], 0],
		[0, 0, v[0]],
		[0, 0, v[1]],
	])


def exp(v):
    """
    Given an 8-vector v, compute a homography H. This is the matrix exponential of hat(v).
    """
    v = np.asarray(v)
    assert np.shape(v) == (8,), 'exp() received shape %s' % str(v.shape)
    return scipy.linalg.expm(hat(v))


def log(h):
    """
    Compute the Lie algebra representation of a homography.
    """
    h = np.asarray(h)
    assert np.shape(h) == (3, 3), 'log() received shape %s' % str(np.shape(h))
    return unhat(scipy.linalg.logm(h))


def perturb_left(h, dh):
    """
    Compute exp(dh)*h where h is a 3x3 homography and dh is 3x3 with trace zero
    """
    return np.dot(exp(dh), h)


def displacement_left(h1, h2):
    """
    Find a vector v such that exp(v) * h1 = h2.
    """
    assert np.shape(h1) == (3, 3), 'log() received shape %s' % str(np.shape(h1))
    assert np.shape(h2) == (3, 3), 'log() received shape %s' % str(np.shape(h1))
    return log(np.dot(h2, np.linalg.inv(h1)))


class Atlas(object):
    """
    Represents a mapping from 3D rotations to a local parameterization for rotations.
    """
    @classmethod
    def perturb(cls, h, delta):
        return perturb_left(h, delta)

    @classmethod
    def displacement(cls, h1, h2):
        return displacement_left(h1, h2)

    @classmethod
    def dof(cls, _):
        return 8
