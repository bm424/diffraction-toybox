from unittest import TestCase

from toybox.symmetry import Rotation, Reflection
from toybox.symmetry import parse_hermann_mauguin


class TestParseHermannMauguin(TestCase):

    def test_full_expression(self):
        operations = parse_hermann_mauguin("6m")
        self.assertEqual(operations[0], Rotation.from_symmetry(6))
        self.assertEqual(operations[1], Reflection.from_orientation(0))

    def test_rotation_expression(self):
        operations = parse_hermann_mauguin(3)
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0], Rotation.from_symmetry(3))

    def test_reflection_expression(self):
        operations = parse_hermann_mauguin("mm")
        self.assertEqual(len(operations), 2)
        self.assertEqual(operations[0], Reflection.from_orientation(0))
        self.assertEqual(operations[1], Reflection.from_orientation(90))

    def test_bad_expression(self):
        with self.assertRaises(ValueError):
            parse_hermann_mauguin("m4")

    def test_invalid_expression(self):
        with self.assertRaises(ValueError):
            parse_hermann_mauguin("spam")


