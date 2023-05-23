Python binding for the [Cpufit library](https://github.com/gpufit/Gpufit) which implements Levenberg Marquardt curve fitting in CUDA

Requirements

- Windows
- Python 2 or 3 with NumPy

Installation

Currently the wheel file has to be installed locally.

If NumPy is not yet installed, install it using pip from the command line

pip install numpy

Then install pyCpufit from the local folder via:

pip install --no-index --find-links=LocalPathToWheelFile pyCpufit

Examples

See project-root/examples/python folder.
