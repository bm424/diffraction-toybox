from math import degrees

import numpy as np
from copy import copy
from symmetry.matrices import get_rotation_matrix, get_reflection_matrix
from toybox.tools import equivalent, sort_points, clean_points

IDENTITY = np.array([
    [1., 0.],
    [0., 1.]
])


class BaseOperator:

    def __init__(self, matrix=IDENTITY, description="Unspecified operation"):
        """Create an operator from a transformation matrix.

        Parameters
        ----------
        matrix : array_like
            The transformation matrix describing the operation
        description : str
            A short description of the operator

        """
        self.matrix = np.array(matrix)
        self._description = description

    @property
    def description(self):
        return self._description

    def apply(self, points):
        """Applies the operator to each of the points in `points`.

        Parameters
        ----------
        points : array_like
            (n_points, n_dimensions + 1)
            A series of n-dimensional points, where the last column is reserved
            for point intensity.

        Returns
        -------
        transformed_points : ndarray
            (n_points, n_dimensions + 1)
            The new position of the points.

        """
        transformed_points = np.dot(self.matrix, points.T).T
        transformed_points = clean_points(transformed_points)
        return transformed_points


    def __mul__(self, other):
        new_matrix = np.dot(self.matrix, other.matrix)
        new_description = self.description + "\n" + other.description
        return BaseOperator(new_matrix, new_description)

    def __eq__(self, other):
        if np.all(np.isclose(self.matrix, other.matrix)):
            return True
        else:
            return False

    def __repr__(self):
        return "{}:\n{}".format(self.description, np.round(self.matrix), 2)


class Rotation(BaseOperator):

    _angle = 0.

    @property
    def description(self):
        return "Rotation through {} degrees".format(self.angle)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self.matrix = get_rotation_matrix(angle)
        self._angle = angle

    @staticmethod
    def from_angle(angle=0., angle_type="degree"):
        if angle_type == "radian":
            angle = degrees(angle)
        matrix = get_rotation_matrix(angle)
        rotation = Rotation(matrix)
        rotation.angle = angle
        return rotation

    @staticmethod
    def from_symmetry(symmetry=1):
        angle = 360/symmetry
        matrix = get_rotation_matrix(angle)
        rotation = Rotation(matrix)
        rotation.angle = angle
        return rotation


class Reflection(BaseOperator):

    _orientation = 0.

    @property
    def description(self):
        return "Reflection about {} degrees".format(self.orientation)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self.matrix = get_reflection_matrix(orientation)
        self._orientation = orientation

    @staticmethod
    def from_orientation(orientation=90., angle_type="degree"):
        if angle_type == "radian":
            orientation = degrees(orientation)
        matrix = get_reflection_matrix(orientation)
        reflection = Reflection(matrix)
        reflection.orientation = orientation
        return reflection


def propagate(points, *operators, max_iter=1000):
    new_points = points.copy()
    n = 0
    for operator in operators:
        while True:
            n += 1
            if n > max_iter:
                return new_points
            old_points = copy(new_points)
            intensities = new_points[:, -1].reshape(-1, 1)
            transformed_coordinates = operator.apply(new_points[:, :-1].astype(float))
            transform = np.hstack((transformed_coordinates, intensities))
            new_points = np.vstack((new_points, transform))
            new_points = np.vstack({tuple(row) for row in new_points})
            if equivalent(new_points, old_points):
                break
    return sort_points(new_points)