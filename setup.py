from distutils.core import setup

setup(
    name='diffraction-toybox',
    version='0.1',
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
