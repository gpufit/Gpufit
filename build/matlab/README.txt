Matlab binding for the [Gpufit library](https://github.com/gpufit/Gpufit) which implements Levenberg Marquardt curve fitting in CUDA

Requirements

- A CUDA capable graphics card with a recent Nvidia graphics driver (at least 367.48 / July 2016)
- Windows
- Matlab 32/64bit

Installation

An installation is not necessary. However, this path must be part of the Matlab path. Use `addpath` if necessary.

Examples

See examples folder. The examples are fully functional only from Matlab2014a.

Troubleshooting

A common reason for the error message 'CUDA driver version is insufficient for CUDA runtime version' is an outdated Nvidia graphics driver.