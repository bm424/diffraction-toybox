from unittest import TestCase

import numpy as np

from toybox.tools import *


class TestCheckPoint(TestCase):

    def test_valid_point(self):
        point = (2, 3)
        check_point(point)

    def test_valid_point_with_intensity(self):
        point = (2, 3, 1.)
        check_point(point)

    def test_check_3d_point(self):
        point = (0., 1., 0., 1.)
        with self.assertRaises(ValueError):
            check_point(point)

    def test_check_invalid_point(self):
        point = (0., 'bad value')
        with self.assertRaises(ValueError):
            check_point(point)


class TestCheckPoints(TestCase):

    def test_single_point(self):
        point = ((2, 3, None),)
        check_points(point)


class TestSortPoints(TestCase):

    def test_sort_points(self):
        points = np.array([
            [1., 1.],
            [2., 2.],
            [1., 2.],
        ])
        result = sort_points(points)
        np.testing.assert_array_almost_equal(result, points)

    def test_sort_points_with_zero(self):
        points = np.array([
            [1., 0.],
            [0., 0.]
        ])
        expected = np.array([
            [0., 0.],
            [1., 0.]
        ])
        result = sort_points(points)
        np.testing.assert_array_almost_equal(result, expected)