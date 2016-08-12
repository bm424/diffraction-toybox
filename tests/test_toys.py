import unittest
import numpy as np
from toybox.toys import Points
from toybox.tools import check_point


class TestPoints(unittest.TestCase):

    def setUp(self):
        self.points = np.array([
            (1., 0.,),
            (1., 1.,),
        ])
        self.pattern = Points(self.points)

    def test_create_without_zero(self):
        pattern = Points(self.points, auto_zero=False)
        zero = (0., 0.)
        self.assertTrue(
            np.sum(np.product(pattern.positions == zero, axis=1)) == 0
        )

    def test_create_with_zero(self):
        pattern = Points(self.points, auto_zero=True)
        zero = (0., 0.)
        self.assertTrue(
            np.sum(np.product(pattern.positions == zero, axis=1)) == 1
        )

    def test_append_valid_point(self):
        point = (0., 1.)
        self.pattern.append_point(point)
        self.assertEqual(
            np.sum(np.product(self.pattern.positions == point, axis=1)), 1
        )


    def test_to_square(self):
        expected = np.array([
            [50, 50],
            [90, 50],
            [90, 90]
        ])
        result = self.pattern.to_shape((100, 100))
        np.testing.assert_array_almost_equal(result, expected)

    def test_to_rectangle(self):
        expected = np.array([
            [100, 50],
            [180, 50],
            [180, 90]
        ])
        result = self.pattern.to_shape((200, 100))
        np.testing.assert_array_almost_equal(result, expected)



