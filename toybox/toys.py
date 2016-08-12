from collections import UserList

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from skimage import filters

from symmetry.operators import propagate
from symmetry.parsers import parse_hermann_mauguin
from toybox.tools import check_point, check_points, equivalent


class Points:
    """A series of points related by symmetry.

    Use this class to define where the diffraction points should be in relation
    to one another.

    Attributes
    ----------
    starting_points : ndarray
        (n_points, 2)
        The basis points, which are used to construct the remaining points
        through symmetry.
    symmetry : str, int
        The symmetry of the pattern. `1` indicates no symmetry.
    points : ndarray
        (n_points, 2)
        The points of the pattern with symmetry relations applied. Setting
        `points` will overwrite `starting_points`:

    """

    def __init__(self,
                 points=None,
                 symmetry=1,
                 auto_zero=True,
                 ):
        if points is not None:
            starting_points = check_points(points)
            self.starting_points = np.array(starting_points)
        else:
            self.starting_points = None
        self.symmetry = symmetry

        if auto_zero:
            self.append_point((0., 0.))

    def append_point(self, point):
        """Adds a point to the pattern.

        Parameters
        ----------
        point : array_like
            Coordinates of the point to add.

        """
        point = check_point(point)
        if self.starting_points is None:
            self.starting_points = point
        else:
            self.starting_points = np.vstack((self.starting_points, point))
        return self

    @property
    def points(self):
        operations = parse_hermann_mauguin(self.symmetry)
        return propagate(self.starting_points, *operations)

    @points.setter
    def points(self, points):
        self.starting_points = check_points(points)

    @property
    def positions(self):
        return self.points[:, :2].astype(float)

    @property
    def intensities(self):
        return self.points[:, 2]

    @intensities.setter
    def intensities(self, intensities):
        self.starting_points[:, 2] = intensities

    def to_shape(self, shape, scale=0.8):
        """Scales and translates the points into a bounding box of size `shape`.

        Parameters
        ----------
        shape : tuple
            The shape of the bounding box.
        scale : float
            All the new points will fit within a factor of `scale` from the
            centre to the edge of the bounding box.

        Returns
        -------
        ndarray
            (n_points, 2)
            The transformed points.

        """
        offset = np.array(shape)/2
        scale_factor = scale * offset
        return self.positions * scale_factor + offset

    def __repr__(self):
        return "Array\n-----\nSymmetry: {}\n{}".format(self.symmetry,
                                                       self.points)

    def __eq__(self, other):
        if equivalent(self.points, other.points):
            return True
        else:
            return False


class Pattern(np.ndarray):
    """A class representing a toy pattern.

    Subclassing np.ndarray, this class simply extends the functionality of the
    array.

    """

    @classmethod
    def from_points(cls, points, shape=(100, 100), scale=0.8, blur=1.):
        """Creates a pattern from a set of points.

        Currently only Gaussian peaks are implemented.

        Parameters
        ----------
        points : Points, array_like
            Positions and intensities of the points in the array.
        shape : Shape of the final array.
        scale : float
            Maximum extent of the points. Should be less than 1.
        blur : float
            Level of gaussian blur to apply to the pattern.

        Returns
        -------
        Pattern
            An array simulating a diffraction pattern.

        """
        if not isinstance(points, Points):
            points = Points(points)
        positions = points.to_shape(shape, scale)
        dat = np.zeros(shape)
        x, y = np.mgrid[0: shape[0], 0: shape[1]]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        for position, intensity in zip(positions, points.intensities):
            dat += intensity * multivariate_normal.pdf(pos, mean=position, cov=1)
        dat = filters.gaussian(dat, sigma=blur)
        return dat.view(cls)

    def plot(self, colorbar=False, cmap='gray'):
        plt.imshow(self, interpolation='none', cmap=cmap)
        if colorbar:
            plt.colorbar()















