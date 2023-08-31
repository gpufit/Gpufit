"""
    setup script for pyCpufit

    TODO get version, get meaningful email
"""

from setuptools import setup, find_packages
import os
from io import open # to have encoding as parameter of open on Python >=2.6
import pycpufit.version as vs

if os.name == 'nt':
	lib_ext = '.dll' # library name extension on Windows
elif os.name == 'posix':
	lib_ext = '.so'  # library name extensions on Unix
else:
	raise RuntimeError('OS {} not supported'.format(os.name))

HERE = os.path.abspath(os.path.dirname(__file__))

CLASSIFIERS = ['Development Status :: 5 - Production/Stable',
               'Intended Audience :: End Users/Desktop',
               'Operating System :: Microsoft :: Windows',
               'Topic :: Scientific/Engineering',
               'Topic :: Software Development :: Libraries']

def get_long_description():
    """
    Get the long description from the README file.
    """
    with open(os.path.join(HERE, 'README.txt'), encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    setup(name='pyCpufit',
        version=vs.__version__,
        description='Levenberg Marquardt curve fitting on the CPU',
        long_description=get_long_description(),
        url='https://github.com/gpufit/Gpufit',
        author='M. Bates, A. Przybylski, B. Thiel, J. Dokulil, and J. Keller-Findeisen',
        author_email='a@b.c',
        license='MIT license',
        classifiers=[],
        keywords='Levenberg Marquardt, curve fitting',
        packages=find_packages(where=HERE),
        package_data={'pycpufit': ['*{}'.format(lib_ext)]},
        install_requires=['NumPy>=1.8'],
        zip_safe=False)
