from . import __version__
from distutils.core import setup

setup(
    name='diffraction-toybox',
    version=__version__,
    packages=[
        'tests',
        'toybox',
        'symmetry'
    ],
    url='',
    license='MIT',
    author='bm424',
    author_email='bm424@cam.ac.uk',
    description='A coordinated set of toys for quickly prototyping and experimenting with diffraction-pattern-like data.'
)
