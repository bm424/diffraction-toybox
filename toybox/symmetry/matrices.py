from math import radians, cos, sin
import numpy as np
from numpy.linalg import inv


def rotation_matrix(theta):
    """Generates a 2-d rotation matrix.

    .. math::
        \\begin{pmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\
        \\cos\\theta \\end{pmatrix}


    Parameters
    ----------
    theta : float
        Angle of rotation in degrees.

    Returns
    -------
    ndarray
        A 2-d rotation matrix.

    """
    theta = radians(theta)
    matrix = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])
    return matrix


def reflection_matrix(orientation=90.):
    """Generates a 2-d reflection matrix.

    .. math::
        \\begin{pmatrix} \\cos 2\\theta & \\sin 2\\theta \\\\ \\sin 2\\theta \\
        & -\\cos 2\\theta \\end{pmatrix}

    Parameters
    ----------
    orientation : float, optional
        Angle line of reflection makes to the x-axis in degrees.

    Returns
    -------
    ndarray
        A 2-d reflection matrix.

    """
    orientation = radians(orientation)
    return np.array([
        [cos(2 * orientation), sin(2 * orientation)],
        [sin(2 * orientation), -cos(2 * orientation)]
    ])


def check_orthogonal(matrix):
    """Checks if `matrix` is orthogonal by equating the inverse and transpose.

    Parameters
    ----------
    matrix : array_like
        An array representing a matrix.

    Returns
    -------
    bool
        True if the matrix is orthogonal, False otherwise.

    """
    matrix_t = np.transpose(matrix)
    matrix_i = inv(matrix)

    if not np.all(np.isclose(matrix_t, matrix_i)):
        return False
    return True


def get_point_group_transformation_matrix(matrix_type, rotation_or_matrix):
    """Generates a matrix from the input

    Parameters
    ----------
    matrix_type : function
        Any function able to return an orthogonal matrix.
    rotation_or_matrix : int, float, array_like
        An object to convert into a matrix. If an int or a float, this will be
        passed to the `matrix_type` creator. If a matrix, it will be checked
        for orthogonality.

    Returns
    -------
    ndarray
        A matrix representing the specified transformation.

    """
    matrix = None
    if matrix_type == "rotation":
        matrix_type = rotation_matrix
    elif matrix_type == "reflection":
        matrix_type = reflection_matrix

    if isinstance(rotation_or_matrix, float) or isinstance(rotation_or_matrix,
                                                           int):
        matrix = matrix_type(rotation_or_matrix)
    elif isinstance(rotation_or_matrix, np.ndarray):
        matrix = rotation_or_matrix
    if not check_orthogonal(matrix):  # Simple check to see if the matrix is a valid rotation matrix.
        raise ValueError(
            "Not a valid point group transformation matrix (non-orthogonal).")
    assert matrix is not None
    return matrix


def get_rotation_matrix(rotation):
    return get_point_group_transformation_matrix("rotation", rotation)


def get_reflection_matrix(orientation):
    return get_point_group_transformation_matrix("reflection", orientation)
