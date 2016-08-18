import unittest

import numpy as np
from toybox.toys.crystals import BiCrystal
from toybox.toys.core import Points


class TestPoints(unittest.TestCase):

    def setUp(self):
        self.points = np.array([
            (1., 0.,),
        ])
        self.pattern = Points(self.points)

    def test_create_without_zero(self):
        pattern = Points(self.points, auto_zero=False)
        zero = (0., 0.)
        self.assertTrue(
            np.sum(np.product(pattern.positions == zero, axis=1)) == 0
        )

    def test_create_with_zero(self):
        pattern = Points(auto_zero=True)
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
            [100, 50],
        ])
        result = self.pattern.to_shape((100, 100))
        np.testing.assert_array_almost_equal(result, expected)

    def test_to_rectangle(self):
        expected = np.array([
            [100, 50],
            [200, 50],
        ])
        result = self.pattern.to_shape((200, 100))
        np.testing.assert_array_almost_equal(result, expected)


class TestBiCrystal(unittest.TestCase):

    def setUp(self):
        self.pattern1 = np.zeros((100, 100))
        self.pattern2 = np.ones((100, 100))
        self.bicrystal = BiCrystal(self.pattern1, self.pattern2)

    def test_init(self):
        bicrystal = BiCrystal(self.pattern1, self.pattern2)
        self.assertTrue(np.all(bicrystal[0] == self.pattern1))

    def test_len(self):
        del self.bicrystal[3]
        self.assertEqual(len(self.bicrystal), 10)

    def test_insert(self):
        self.bicrystal.insert(5, 1)
        self.assertTrue(np.all(self.bicrystal[5] == self.pattern2))

    def test_set_profile_too_high(self):
        with self.assertRaises(ValueError):
            self.bicrystal.profile = np.linspace(0, 2, 11)

    def test_set_profile_too_low(self):
        with self.assertRaises(ValueError):
            self.bicrystal.profile = np.linspace(-1, 1, 11)

    def test_set_profile_element_too_low(self):
        with self.assertRaises(ValueError):
            self.bicrystal[0] = -1

    def test_set_profile_element_too_high(self):
        with self.assertRaises(ValueError):
            self.bicrystal[1] = 1.2





