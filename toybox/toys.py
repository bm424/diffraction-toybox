import collections

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
        (n_points, 3)
        The basis points, which are used to construct the remaining points
        through symmetry.
    symmetry : str, int
        The symmetry of the pattern. `1` indicates no symmetry.
    points : ndarray
        (n_points, 3)
        The points of the pattern with symmetry relations applied. Setting
        `points` will overwrite `starting_points`:

    """

    def __init__(self,
                 points=None,
                 symmetry=1,
                 auto_zero=True,
                 ):
        """

        Parameters
        ----------
        points : array_like
            (n_points, 3)
            Initial points, in the format (x, y, intensity)
        symmetry : int, str
            Symmetry to apply to the points.
        auto_zero : bool
            If True, automatically appends (0., 0., None) to the points.

        """

        if points is not None:
            starting_points = check_points(points)
            self.starting_points = np.array(starting_points)
        else:
            self.starting_points = None
        if auto_zero:
            self.append_point((0., 0.))
        self.symmetry = symmetry

    def append_point(self, point):
        """Adds a point to the pattern.

        Parameters
        ----------
        point : array_like
            Coordinates of the point to add.

        """
        point = check_point(point)
        if self.starting_points is None:
            self.starting_points = np.array(point).reshape(1, -1)
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

    def to_shape(self, shape, scale=1.0):
        """Scales and translates the points into a bounding box of size `shape`.

        Parameters
        ----------
        shape : tuple
            The shape of the bounding box.
        scale : float, optional
            All the new points will fit within a factor of `scale` from the
            centre to the edge of the bounding box. Defaults to 1.0.

        Returns
        -------
        ndarray
            (n_points, 2)
            The transformed points.

        """
        offset = np.array(shape)/2
        scale_factor = scale * offset
        distance = np.nanmax(np.sqrt(np.sum(np.square(self.positions), axis=1)))
        return (self.positions/distance) * scale_factor + offset

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
    def from_points(cls, points, shape=(100, 100), scale=1.0, blur=1.):
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


class BiCrystal(collections.MutableSequence):

    def __init__(self, pattern1, pattern2, profile=np.linspace(0, 1, 11)):
        self.pattern_1 = pattern1
        self.pattern_2 = pattern2
        self.profile = profile

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, profile):
        if np.max(profile) > 1 or np.min(profile) < 0:
            raise ValueError("Profile must be between 0 and 1.")
        self._profile = profile

    @property
    def profile_i(self):
        return 1. - self.profile

    @property
    def patterns(self):
        return np.array([p * self.pattern_2 + q * self.pattern_1 for p, q in zip(self.profile, self.profile_i)])

    def __len__(self):
        return len(self.profile)

    def __getitem__(self, item):
        return self.patterns[item].view(Pattern)

    def __setitem__(self, key, value):
        if value > 1 or value < 0:
            raise ValueError("Profile must be between 0 and 1.")
        self.profile[key] = value

    def __delitem__(self, key):
        self.profile = np.delete(self.profile, key, None)

    def insert(self, index, value):
        self.profile = np.insert(self.profile, index, value)


















