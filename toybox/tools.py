import numpy as np


def check_point(point):
        """Validates a 2-d point with an intensity.

        Parameters
        ----------
        point : array_like
            (x, y, [intensity])
            Coordinates of the point to add. If optional `intensity` is not
            supplied, it will be set to `None`.

        Returns
        -------
        point : :obj:tuple of :obj:float
            The same point converted to floats, assuming no errors were raised.

        """
        if len(point) != 2:
            intensity = point[-1]
            point = point[:-1]
        else:
            intensity = None
        if len(point) != 2:
            raise ValueError("Coordinates must be two-dimensional.")
        try:
            point = tuple([float(p) for p in point])  # Convert everything to floats.
        except ValueError:
            raise ValueError("Coordinates must be convertible to floats.")
        return point + tuple((intensity,))


def check_points(points):
    """Calls :func:`check_point` for every point passed.

    Parameters
    ----------
    points : array_like
        (n_points, 3)
        The points to be checked: (x, y, intensity)
    """
    checked_points = []
    for point in points:
        checked_points.append(check_point(point))
    return np.array(checked_points)


def equivalent(points1, points2):
    """Checks whether two sets of points are equivalent by comparing rows.

    The points are converted into arrays of strings. If every string in array 1
    is also in array 2 and vice-versa, the sets of points are considered to be
    equivalent.

    Parameters
    ----------
    points1, points2 : array_like
        (n_points, n_dimensions)
        A series of points.

    Returns
    -------
    bool
        True if all points in `points1` are in `points2`. Otherwise False.

    References
    ----------
    http://stackoverflow.com/a/16240957

    """
    points_string_1 = np.array([str(point) for point in points1],
                               dtype=str).reshape(points1.shape[0])
    points_string_2 = np.array([str(point) for point in points2],
                               dtype=str).reshape(points2.shape[0])
    if np.all(np.in1d(points_string_1, points_string_2)) and np.all(
            np.in1d(points_string_2, points_string_1)):
        return True
    else:
        return False


def sort_points(points):
    """Sorts points first by argument, then by modulus.

    Parameters
    ----------
    points : array_like
        (n_points, 3)
        The points to be sorted: (x, y, intensity)

    Returns
    -------
    points_sorted : :class:`numpy.ndarray`
        (n_points, 3)
        The sorted points.

    """
    positions = points[:, :2].astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        tangents = np.nan_to_num(positions[:, 1]/positions[:, 0])
    arguments = np.arctan(tangents)
    moduli = np.sqrt(np.sum(np.square(positions), axis=1))
    inds = np.lexsort((moduli, arguments))
    points_sorted = points[inds]
    return points_sorted


def clean_points(points):
    """Rounds point coordinates to 9 decimal places. Suppresses near-zeros."""
    points[np.isclose(points, 0.)] = 0  # Dot product has a bad habit of adding small amounts to zero values. Suppress it here.
    points = np.round(points, 9)
    return points
