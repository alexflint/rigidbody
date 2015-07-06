import unittest
import numpy as np
from numericaltesting import assert_arrays_almost_equal

from . import arithmetic

class ArithmeticTest(unittest.TestCase):
	def test_normalized(self):
		a = [2., 0., 0.]
		assert_arrays_almost_equal(arithmetic.normalized(a), [1., 0., 0.])

	def test_pr(self):
		a = [4., 6., 2.]
		assert_arrays_almost_equal(arithmetic.pr(a), [2., 3.])

	def test_unpr(self):
		a = [1., 2.]
		assert_arrays_almost_equal(arithmetic.unpr(a), [1., 2., 1.])

	def test_unreduce(self):
		a = [1, 2]
		mask = [False, True, False, True, False]
		assert_arrays_almost_equal(arithmetic.unreduce(a, mask), [0, 1, 0, 2, 0])

	def test_unreduce_2d(self):
		a = [[1, 2], [3, 4]]
		mask = [True, False, True]
		expected = [
			[1, 0, 2],
			[0, 0, 0],
			[3, 0, 4]]
		assert_arrays_almost_equal(arithmetic.unreduce_2d(a, mask), expected)

	def test_dots(self):
		a = np.eye(2)
		b = np.array([[1., 2.], [3., 4.]])
		assert_arrays_almost_equal(arithmetic.dots(a, b, a), b)

	def test_sumsq(self):
		a = [1, 2]
		assert_arrays_almost_equal(arithmetic.sumsq(a), 5)

	def test_skew(self):
		a = [1, 2, 3]
		expected = [
			[0, -3, 2],
			[3, 0, -1],
			[-2, 1, 0]]
		assert_arrays_almost_equal(arithmetic.skew(a), expected)

	def test_unit(self):
		assert_arrays_almost_equal(arithmetic.unit(1, 5), [0, 1, 0, 0, 0])

	def test_min_med_max(self):
		a = [1, 2, 3, 4, 5]
		assert_arrays_almost_equal(arithmetic.minmedmax(a), [1, 3, 5])




if __name__ == '__main__':
	unittest.main()
