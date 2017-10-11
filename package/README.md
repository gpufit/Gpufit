# Creating a binary package

The binary package bundles different builds outputs into a single distributable binary package containing the Gpufit dll,
the performance comparison example, the Matlab bindings and the Python bindings.

## Calling the script

create_package.bat %1 %2 %3

with 

- %1 is the BUILD_BASE_PATH (the path containing the various (see below) CMake generated Visual Studio projects)

- %2 is the VERSION (e.g. 1.0.0)

- %3 is the SOURCE_BASE_PATH (the path containing the sources)

The output is a folder (BUILD_BASE_PATH/Gpufit-VERSION) which is also zipped if 7-Zip is available.

## Requirements

Note: The script has no way of checking that the requirements are fulfilled!

See also [Build from sources](http://Gpufit.readthedocs.io/en/latest/installation.html#build-from-sources) for instructions.

CMake

- CUDA_ARCHITECTURE must be set to All (it is by default)

- CUDA toolkit 8.0/9.0 is used for all builds (must be installed before)

- Build directory for MSVC14 Win64 is BUILD_BASE_PATH/VC14x64-8.0

- Build directory for MSVC14 Win32 is BUILD_BASE_PATH/VC14x32-8.0

- Matlab and Python must be available

Build

- Configuration RelWithDebInfo is used for all builds!

- With MSVC14 Win64 build target PYTHON_WHEEL, MATLAB_GPUFIT_PACKAGE and the Gpufit_Cpufit_Performance_Comparison example

- With MSVC14 Win32 build target PYTHON_WHEEL, MATLAB_GPUFIT_PACKAGE and the Gpufit_Cpufit_Performance_Comparison example

Documentation

- An up-to-date version of the documentation must exist at SOURCE_BASE_PATH\docs\_build\latex\Gpufit.pdf (must be created before).

## Setting the version number

Unfortunately the version has to be updated in various places.

- CmakeLists.txt (project( Gpufit VERSION 1.0.0 ))
- docs/conf.py (release = u'1.0.0')
- docs/epilog.txt (.. |GF_version| replace:: 1.0.0)
- Gpufit/matlab/gpufit_version.m 
- Gpufit/python/pygpufit/version.py
- calling the packaging script (create_package.bat %1 1.0.0 %3)
- package/sdk_readme.txt, also CUDA version inside
