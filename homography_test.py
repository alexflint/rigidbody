import unittest
import numpy as np
from manifolds import numeric_jacobian
from numericaltesting import assert_arrays_almost_equal

from . import homography


class HomographyTest(unittest.TestCase):
    def test_hat(self):
        np.random.seed(0)
        w = np.random.randn(8)
        m = homography.hat(w)
        m_expected = sum(gi * wi for gi, wi in zip(homography.generators, w))
        np.testing.assert_array_almost_equal(m_expected, m)

    def test_unhat(self):
        np.random.seed(0)
        w = np.random.randn(8)
        m = homography.hat(w)
        np.testing.assert_array_almost_equal(w, homography.unhat(m))

    def test_exp(self):
        np.random.seed(0)
        w = np.random.randn(8)
        h = homography.exp(w)
        self.assertAlmostEqual(np.linalg.det(h), 1.)

    def test_jacobian(self):
        np.random.seed(0)
        p = np.random.randn(3)

        print(homography.exp(np.zeros(8)))

        j_numerical = numeric_jacobian(lambda x: np.dot(homography.exp(x), p),
                                       np.zeros(8))
        j_analytic = homography.multiply_generators(p).T
        np.testing.assert_array_almost_equal(j_numerical, j_analytic)
