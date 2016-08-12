import unittest
from math import sqrt, sin, cos, radians, pi

from unittest import TestCase

from random import random
import numpy as np

from symmetry.operators import BaseOperator, Rotation, Reflection, propagate
from toybox.tools import equivalent

ROOT2 = sqrt(2)
IDENTITY = np.array([
    [1, 0],
    [0, 1]
])


class TestRotation(unittest.TestCase):
    def setUp(self):
        self.points = np.array([
            [1., 0.],
            [1., 1.]
        ])

    def test_creation_from_angle(self):
        angle = 90
        correct_answer = np.array([
            [0., -1.],
            [1., 0.]
        ])
        rotation = Rotation.from_angle(angle)
        self.assertEqual(rotation.angle, angle)
        np.testing.assert_array_almost_equal(rotation.matrix, correct_answer)

    def test_creation_from_symmetry(self):
        symmetry = 2
        correct_answer = np.array([
            [-1, 0],
            [0, -1]
        ])
        rotation = Rotation.from_symmetry(symmetry)
        self.assertEqual(rotation.angle, 180)
        np.testing.assert_array_almost_equal(rotation.matrix, correct_answer)


    def test_apply_45(self):
        correct_answer = np.array([
            [ROOT2 / 2, ROOT2 / 2],
            [0, ROOT2]
        ])
        new_points = Rotation.from_angle(45).apply(self.points)
        np.testing.assert_array_almost_equal(new_points, correct_answer)

    def test_apply_90(self):
        correct_answer = np.array([
            [0., 1.],
            [-1., 1.]
        ])
        new_points = Rotation.from_angle(90).apply(self.points)
        np.testing.assert_array_almost_equal(new_points, correct_answer)

    def test_angle_setting(self):
        correct_answer = np.array([
            [0, -1],
            [1, 0]
        ])
        rotation = Rotation()
        rotation.angle = 90
        np.testing.assert_array_almost_equal(rotation.matrix, correct_answer)


class TestReflection(unittest.TestCase):
    def setUp(self):
        self.points = np.array([
            [1., 0.],
            [1., 1.]
        ])

    def test_creation_from_orientation(self):
        orientation = 90
        correct_answer = np.array([
            [-1., 0.],
            [0., 1.]
        ])
        reflection = Reflection.from_orientation(orientation)
        self.assertEqual(reflection.orientation, orientation)
        np.testing.assert_array_almost_equal(reflection.matrix, correct_answer)


    def test_apply_45(self):
        correct_answer = np.array([
            [0., 1.],
            [1., 1.]
        ])
        new_points = Reflection.from_orientation(45).apply(self.points)
        np.testing.assert_array_almost_equal(new_points, correct_answer)

    def test_apply_90(self):
        correct_answer = np.array([
            [-1., 0.],
            [-1., 1.]
        ])
        new_points = Reflection.from_orientation(90).apply(self.points)
        np.testing.assert_array_almost_equal(new_points, correct_answer)

    def test_orientation_setting(self):
        correct_answer = np.array([
            [-1, 0],
            [0, 1]
        ])
        reflection = Reflection()
        reflection.orientation = 90
        np.testing.assert_array_almost_equal(reflection.matrix, correct_answer)

    def test_creation_from_angle(self):
        angle = 90
        correct_answer = np.array([
            [-1., 0.],
            [0., 1.]
        ])
        reflection = Reflection.from_orientation(angle)
        self.assertEqual(reflection.orientation, angle)
        np.testing.assert_array_almost_equal(reflection.matrix, correct_answer)


class TestComposition(unittest.TestCase):
    def setUp(self):
        self.rotation_angle = 360. * random()
        self.orientation_angle = 360 * random()
        self.rotation = Rotation.from_angle(self.rotation_angle)
        self.reflection = Reflection.from_orientation(self.orientation_angle)

    def test_composition(self):
        self.assertIsInstance(self.rotation * self.reflection, BaseOperator)

    def test_double_rotation(self):
        expected = Rotation.from_angle(2 * self.rotation_angle)
        composition = self.rotation * self.rotation
        self.assertEqual(composition, expected)

    def test_double_reflection(self):
        expected = BaseOperator(IDENTITY)
        composition = self.reflection * self.reflection
        self.assertEqual(composition, expected)

    def test_two_different_reflections(self):
        second_orientation = 360 * random()
        second_reflection = Reflection.from_orientation(second_orientation)
        expected = Rotation.from_angle(
            2 * (self.orientation_angle - second_orientation))
        composition = self.reflection * second_reflection
        self.assertEqual(composition, expected)

    def test_reflection_then_rotation(self):
        expected = Reflection.from_orientation(
            self.orientation_angle + 0.5 * self.rotation_angle)
        composition = self.rotation * self.reflection
        self.assertEqual(composition, expected)

    def test_rotation_then_reflection(self):
        expected = Reflection.from_orientation(
            self.orientation_angle - 0.5 * self.rotation_angle)
        composition = self.reflection * self.rotation
        self.assertEqual(composition, expected)


class TestEquivalent(TestCase):
    def test_equivalent(self):
        pts1 = np.array([
            [1, 0],
            [0, 1]
        ])
        pts2 = np.array([
            [0, 1],
            [1, 0]
        ])
        self.assertTrue(equivalent(pts1, pts2))

    def test_not_equivalent(self):
        pts1 = np.array([
            [1, 0],
            [0, 1]
        ])
        pts2 = np.array([
            [2, 0],
            [0, 1]
        ])
        self.assertFalse(equivalent(pts1, pts2))


class TestPropagate(TestCase):
    def setUp(self):
        self.pts1 = np.array([
            [1., 0., None],
            [1., 1., 1.]
        ])

    def test_propagate_rotation(self):
        expected = np.array([
            [1., 0., None],
            [0., 1., None],
            [-1., 0., None],
            [0., -1., None]
        ])
        rotation = Rotation.from_angle(90)
        result = propagate(self.pts1[0:1], rotation)
        self.assertTrue(equivalent(result, expected),
                        msg="Expected {}, got {}".format(expected, result))

    def test_propagate_multiple(self):
        angle = 120
        expected = np.array([
            [1, 0],
            [cos(pi/3), sin(pi/3)],
            [cos(2*pi/3), sin(2*pi/3)],
            [-1, 0],
            [cos(4*pi/3), sin(4*pi/3)],
            [cos(5*pi/3), sin(5*pi/3)]
        ])
        rotation = Rotation.from_angle(angle)
        reflection = Reflection.from_orientation(90)
        result = np.round(propagate(self.pts1[0:1], rotation, reflection)[:, :2].astype(float), 7)
        self.assertTrue(equivalent(result, expected),
                        msg="Expected {},\ngot {}".format(expected, result))

    def test_propagate_intensities(self):
        expected = np.array([
            [1., 0., None],
            [0., 1., None],
            [-1., 0., None],
            [0., -1., None],
            [1., 1., 1.],
            [-1., 1., 1.],
            [-1., -1., 1.],
            [1., -1., 1.]
        ])
        rotation = Rotation.from_symmetry(4)
        result = propagate(self.pts1, rotation)
        self.assertTrue(equivalent(result, expected),
                        msg="Expected {},\ngot {}".format(expected, result))
