Python binding for the [Gpufit library](https://github.com/gpufit/Gpufit) which implements Levenberg Marquardt curve fitting in CUDA

Requirements

- A CUDA capable graphics card with a recent Nvidia graphics driver (at least 367.48 / July 2016)
- Windows
- Python 2 or 3 with NumPy

Installation

Currently the wheel file has to be installed locally.

If NumPy is not yet installed, install it using pip from the command line

pip install numpy

Then install pyGpufit from the local folder via:

pip install --no-index --find-links=LocalPathToWheelFile pyGpufit

Examples

See examples folder.

Troubleshooting

A common reason for the error message 'CUDA driver version is insufficient for CUDA runtime version' is an outdated Nvidia graphics driver.