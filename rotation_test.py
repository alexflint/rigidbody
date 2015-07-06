import unittest
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from manifolds import numeric_jacobian
from numericaltesting import assert_arrays_almost_equal

from . import rotation


def angular_velocity_global_numerical(r0, r1, delta_t):
    return rotation.log(np.dot(r0.T, r1)) / delta_t


def angular_velocity_local_numerical(r0, r1, delta_t):
    return rotation.log(np.dot(r1, r0.T)) / delta_t


class RotationTest(unittest.TestCase):
    def test_angular_velocity_from_axisangle_rates(self):
        np.random.seed(0)

        # create three splines
        timestamps = np.linspace(0, 10, 10)
        controls = np.random.rand(3, len(timestamps))
        fs = [InterpolatedUnivariateSpline(timestamps, xs) for xs in controls]

        t = 0
        t0 = t - 1e-8
        t1 = t + 1e-8
        r = rotation.exp([f(t) for f in fs])
        r0 = rotation.exp([f(t0) for f in fs])
        r1 = rotation.exp([f(t1) for f in fs])

        w_numerical_global = angular_velocity_global_numerical(r0, r1, t0-t1)
        w_numerical_local = angular_velocity_local_numerical(r0, r1, t0-t1)

        ff = np.array([f(t) for f in fs])
        dfdt = np.array([f.derivative()(t) for f in fs])
        w_analytic_global = rotation.angular_velocity_from_axisangle_rates(ff, dfdt)
        w_analytic_local = np.dot(r, w_analytic_global)

        assert_arrays_almost_equal(w_numerical_global, w_analytic_global, xlabel='numerical', ylabel='analytic')
        assert_arrays_almost_equal(w_numerical_local, w_analytic_local, xlabel='numerical', ylabel='analytic')

    def test_angular_velocity_from_axisangle_rates_at_zero(self):
        np.random.seed(0)

        # create three splines
        timestamps = np.linspace(0, 10, 10)
        controls = np.zeros((3, len(timestamps)))
        fs = [InterpolatedUnivariateSpline(timestamps, xs) for xs in controls]

        t = 0
        t0 = t - 1e-8
        t1 = t + 1e-8
        r = rotation.exp([f(t) for f in fs])
        r0 = rotation.exp([f(t0) for f in fs])
        r1 = rotation.exp([f(t1) for f in fs])

        w_numerical_global = angular_velocity_global_numerical(r0, r1, t0-t1)
        w_numerical_local = angular_velocity_local_numerical(r0, r1, t0-t1)

        ff = np.array([f(t) for f in fs])
        dfdt = np.array([f.derivative()(t) for f in fs])
        w_analytic_global = rotation.angular_velocity_from_axisangle_rates(ff, dfdt)
        w_analytic_local = np.dot(r, w_analytic_global)

        assert_arrays_almost_equal(w_numerical_global, w_analytic_global, xlabel='numerical', ylabel='analytic')
        assert_arrays_almost_equal(w_numerical_local, w_analytic_local, xlabel='numerical', ylabel='analytic')

    def test_exp_jacobian(self):
        np.random.seed(0)
        x0 = np.random.randn(3)
        j_numerical = numeric_jacobian(lambda x: rotation.exp(x), x0,
                                       output_atlas=rotation.RotationAtlas)
        j_analytic = rotation.exp_jacobian(x0)
        np.testing.assert_array_almost_equal(j_numerical, j_analytic)


    def test_displacement_left_jacobian_wrt_lhs(self):
        np.random.seed(0)
        r1 = rotation.exp(np.random.randn(3))
        r2 = rotation.exp(np.random.randn(3))
        j_numerical = numeric_jacobian(lambda r: rotation.displacement_left(r, r2),
                                       r1,
                                       atlas=rotation.RotationAtlas)
        j_analytic = rotation.displacement_left_jacobian_wrt_lhs(r1, r2)
        assert_arrays_almost_equal(j_numerical, j_analytic, xlabel='analytic', ylabel='numerical')


if __name__ == '__main__':
    np.random.seed(123)
    unittest.main()
