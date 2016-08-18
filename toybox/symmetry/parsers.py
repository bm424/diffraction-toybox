import re

from toybox.symmetry.operators import Rotation, Reflection


def parse_hermann_mauguin(symmetry):
    """Converts Hermann-Mauguin notation to symmetry operators.

    Parameters
    ----------
    symmetry : str, int
        Symmetry expressed in Hermann-Mauguin (also known as International)
        notation

    Returns
    -------
    operation : list of operators
        A list containing the required symmetry operations.

    Examples
    --------
    >>> parse_hermann_mauguin("3m")
    [Rotation through 120.0 degrees:
    [[-0. -1.]
     [ 1. -0.]], Reflection about 0 degrees:
    [[ 1.  0.]
     [ 0. -1.]]]

    """
    symmetry = str(symmetry)
    operation = []
    m = re.match(r"^(?P<rotation>\d?)(?P<reflection>m{,2})$", symmetry)
    if m:
        if m.group('rotation'):
            operation.append(Rotation.from_symmetry(int(m.group('rotation'))))
        for r, orientation in zip(range(len(m.group('reflection'))), (0, 90)):
            operation.append(Reflection.from_orientation(orientation))
        return operation
    else:
        raise ValueError("Invalid symmetry: {}".format(symmetry))



