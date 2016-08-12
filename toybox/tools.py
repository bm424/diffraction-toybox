import numpy as np


def check_point(point):
        """Validates a 2-d point with an intensity.

        Parameters
        ----------
        point :

        Returns
        -------
        point : tuple of float

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
    checked_points = []
    for point in points:
        checked_points.append(check_point(point))
    return np.array(checked_points)


def equivalent(points1, points2):
    """Checks whether two sets of points are equivalent by comparing rows.

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
    if np.all(np.in1d(
            np.array([str(point) for point in points1], dtype=str).reshape(points1.shape[0]),
            np.array([str(point) for point in points2], dtype=str).reshape(points2.shape[0])
    )):
        return True
    else:
        return False


def sort_points(points):
    positions = points[:, :2].astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        tangents = np.nan_to_num(positions[:, 1]/positions[:, 0])
    arguments = np.arctan(tangents)
    moduli = np.sqrt(np.sum(np.square(positions), axis=1))
    inds = np.lexsort((moduli, arguments))
    points_sorted = points[inds]
    return points_sorted


def clean_points(points):
    points[np.isclose(points, 0.)] = 0  # Dot product has a bad habit of adding small amounts to zero values. Suppress it here.
    points = np.round(points, 9)
    return points
