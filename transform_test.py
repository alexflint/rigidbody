import unittest
import numpy as np
from numericaltesting import assert_arrays_almost_equal

from .transform import SO3, SE3
from . import rotation


class SO3Test(unittest.TestCase):
    def test_identity(self):
        r = SO3.identity()
        assert_arrays_almost_equal(r.matrix, np.eye(3))

    def test_from_tangent(self):
        r = SO3.from_tangent([.1, .2, .3])
        assert_arrays_almost_equal(r.matrix, rotation.exp([.1, .2, .3]))

    def test_inverse(self):
        r = SO3.from_tangent([.1, .2, .3])
        assert_arrays_almost_equal(r.inverse().matrix, r.matrix.T)

    def test_log(self):
        r = SO3.from_tangent([.1, .2, .3])
        assert_arrays_almost_equal(r.log(), [.1, .2, .3])

class SE3Test(unittest.TestCase):
    def test_identity(self):
        r = SE3.identity()
        assert_arrays_almost_equal(r.matrix[:3, :3], np.eye(3))
        assert_arrays_almost_equal(r.matrix[:3, 3], [0, 0, 0])
        assert_arrays_almost_equal(r.matrix[3], [0, 0, 0, 1])

    def test_from_tangent(self):
        r = SE3.from_tangent([.1, .2, .3, -1, -2, -3])
        assert_arrays_almost_equal(r.orientation, rotation.exp([.1, .2, .3]))
        assert_arrays_almost_equal(r.position, [-1, -2, -3])

    def test_inverse(self):
        r = SE3.from_tangent([.1, .2, .3, -1, -2, -3])
        rinv = SE3(r.orientation.T, -np.dot(r.orientation, r.position))
        assert_arrays_almost_equal(r.inverse().matrix, rinv.matrix)

    def test_log(self):
        r = SE3.from_tangent([.1, .2, .3, -1, -2, -3])
        assert_arrays_almost_equal(r.log(), [.1, .2, .3, -1, -2, -3])

    def test_from_matrix(self):
        r = rotation.exp([.1, .2, .3])
        t = np.array([1., 2., 3.])
        v = SE3.from_matrix(np.hstack((r, t[:, None])))
        assert_arrays_almost_equal(v.orientation, r)
        assert_arrays_almost_equal(v.position, -np.dot(r, t))

    def test_rt(self):
        r = rotation.exp([.1, .2, .3])
        t = np.array([1., 2., 3.])
        v = SE3.from_matrix(np.hstack((r, t[:, None])))
        rr, tt = v.rt
        assert_arrays_almost_equal(r, rr)
        assert_arrays_almost_equal(t, tt)

    def test_rp(self):
        r = rotation.exp([.1, .2, .3])
        p = np.array([1., 2., 3.])
        v = SE3(r, p)
        rr, pp = v.rp
        assert_arrays_almost_equal(r, rr)
        assert_arrays_almost_equal(p, pp)


if __name__ == '__main__':
    unittest.main()


# class SE3(object):
#     """
#     Represents a rigid transform in three dimensions.
#     """
#     DoF = 6

#     class Atlas(object):
#         """
#         Represents an atlas for rigid transforms.
#         """
#         @classmethod
#         def dof(cls, _):
#             return SE3.DoF

#         @classmethod
#         def perturb(cls, pose, tangent):
#             """
#             Evaluate the chart for the given pose at tangent.
#             """
#             assert len(tangent) == SE3.DoF
#             return SE3(rotation.perturb_left(pose.orientation, tangent[:3]), pose.position + tangent[3:])

#         @classmethod
#         def displacement(cls, x1, x2):
#             """
#             Get a vector v such that perturb(x1, v) = x2.
#             """
#             return np.hstack((rotation.log(np.dot(x2.orientation, x1.orientation.T)), x2.position - x1.position))

#     def __init__(self, orientation, position):
#         """
#         Initialize a rigid body transform from a rotation matrix and position vector.
#         """
#         self._orientation = np.asarray(orientation, float)
#         self._position = np.asarray(position, float)

#     @classmethod
#     def identity(cls):
#         """
#         Get the identity transform.
#         """
#         return SE3(np.eye(3), np.zeros(3))

#     @classmethod
#     def from_tangent(cls, v):
#         """
#         Construct a rigid body transform from the tangent space at the identity element.
#         """
#         assert len(v) == SE3.DoF
#         return SE3(rotation.exp(v[:3]), v[3:])

#     @property
#     def orientation(self):
#         """
#         Get the orientation component of this transform.
#         """
#         return self._orientation

#     @orientation.setter
#     def orientation(self, v):
#         """
#         Set the orientation component of this transform.
#         """
#         self._orientation = v

#     @property
#     def position(self):
#         """
#         Get the position component of this transform.
#         """
#         return self._position

#     @position.setter
#     def position(self, v):
#         """
#         Set the position component of this transform.
#         """
#         self._position = v

#     @property
#     def matrix(self):
#         """
#         Get the matrix representation of this transform.
#         """
#         return np.r_[np.c_[self._orientation, -np.dot(self._orientation, self._position)],
#                      np.c_[0., 0., 0., 1.]]

#     @property
#     def rp(self):
#         """
#         Get the (rotation, position) pair for this transform.
#         """
#         return self._orientation, self._position

#     def __mul__(self, rhs):
#         """
#         Multiply this transform with another.
#         """
#         return self.transform(rhs)

#     def transform(self, rhs):
#         """
#         Multiply this transform with another.
#         """
#         if isinstance(rhs, SE3):
#             r1, r2 = self._orientation, rhs._orientation
#             return SE3(np.dot(r1, r2), rhs.position + np.dot(r2.T, self.position))
#         elif isinstance(rhs, np.ndarray):
#             if rhs.shape[-1] == 3:
#                 return np.dot(self._orientation, rhs - self.position)
#             elif rhs.shape[-1] == 4:
#                 return np.dot(self.matrix, rhs)

#     def inverse(self):
#         """
#         Get the inverse of this transform.
#         """
#         return SE3(self._orientation.T, -np.dot(self.orientation, self.position))

#     def __str__(self):
#         """
#         Get a string representation of this transform.
#         """
#         return 'SE3(position=%s, log_rotation=%s)' % (self._position, rotation.log(self._orientation))
