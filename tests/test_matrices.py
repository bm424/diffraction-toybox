from unittest import TestCase

import numpy as np
from toybox.symmetry.matrices import rotation_matrix, reflection_matrix, \
    get_rotation_matrix, get_reflection_matrix


class TestRotationMatrix(TestCase):
    def test_rotation_matrix_0(self):
        correct_answer = np.array([
            [1, 0],
            [0, 1],
        ])
        matrix = rotation_matrix(0)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_rotation_matrix_90(self):
        correct_answer = np.array([
            [0, -1],
            [1, 0],
        ])
        matrix = rotation_matrix(90)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_rotation_matrix_180(self):
        correct_answer = np.array([
            [-1, 0],
            [0, -1],
        ])
        matrix = rotation_matrix(180)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_rotation_matrix_270(self):
        correct_answer = np.array([
            [0, 1],
            [-1, 0],
        ])
        matrix = rotation_matrix(270)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_rotation_matrix_360(self):
        correct_answer = np.array([
            [1, 0],
            [0, 1],
        ])
        matrix = rotation_matrix(360)
        np.testing.assert_array_almost_equal(matrix, correct_answer)


class TestReflectionMatrix(TestCase):
    def test_reflection_matrix_0(self):
        correct_answer = np.array([
            [1, 0],
            [0, -1],
        ])
        matrix = reflection_matrix(0)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_reflection_matrix_45(self):
        correct_answer = np.array([
            [0, 1],
            [1, 0],
        ])
        matrix = reflection_matrix(45)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_reflection_matrix_90(self):
        correct_answer = np.array([
            [-1, 0],
            [0, 1],
        ])
        matrix = reflection_matrix(90)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_reflection_matrix_135(self):
        correct_answer = np.array([
            [0, -1],
            [-1, 0],
        ])
        matrix = reflection_matrix(135)
        np.testing.assert_array_almost_equal(matrix, correct_answer)


class TestGetRotationMatrix(TestCase):

    def test_get_with_int(self):
        correct_answer = np.array([
            [0, -1],
            [1,  0]
        ])
        matrix = get_rotation_matrix(90)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_get_with_float(self):
        correct_answer = np.array([
            [0, -1],
            [1,  0]
        ])
        matrix = get_rotation_matrix(90.)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_get_with_valid_matrix(self):
        rotation = np.array([
            [0, -1],
            [1,  0]
        ])
        matrix = get_rotation_matrix(rotation)
        np.testing.assert_array_almost_equal(matrix, rotation)

    def test_with_invalid_matrix(self):
        rotation = np.array([
            [1, -1],
            [1,  0]
        ])
        with self.assertRaises(ValueError):
            get_rotation_matrix(rotation)


class TestGetReflectionMatrix(TestCase):
    
    def test_get_with_int(self):
        correct_answer = np.array([
            [-1, 0],
            [0,  1]
        ])
        matrix = get_reflection_matrix(90)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_get_with_float(self):
        correct_answer = np.array([
            [-1, 0],
            [0,  1]
        ])
        matrix = get_reflection_matrix(90.)
        np.testing.assert_array_almost_equal(matrix, correct_answer)

    def test_get_with_valid_matrix(self):
        reflection = np.array([
            [1, 0],
            [0, -1]
        ])
        matrix = get_reflection_matrix(reflection)
        np.testing.assert_array_almost_equal(matrix, reflection)

    def test_with_invalid_matrix(self):
        reflection = np.array([
            [1, -1],
            [1,  0]
        ])
        with self.assertRaises(ValueError):
            get_reflection_matrix(reflection)
        

