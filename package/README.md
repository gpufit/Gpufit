# Creating a binary package/release

The binary package bundles different builds outputs into a single distributable binary package containing the Gpufit dll,
the performance comparison example, the Matlab, Python and Java bindings and the documentation.

Follow this step by step recipe to create a Windows binary package.

## Set/Update the version number

Unfortunately the version has to be updated in various places.

- CmakeLists.txt (project( Gpufit VERSION 1.0.0 ))
- docs/conf.py (release = u'1.0.0')
- docs/epilog.txt (.. |GF_version| replace:: 1.0.0)
- Gpufit/matlab/gpufit_version.m 
- Gpufit/python/pygpufit/version.py
- calling the packaging script (create_package.bat %1 1.0.0 %3)
- package/sdk_readme.txt, (also specify CUDA version used for build there)
- Gpufit/java/gpufit/build.gradle (version `1.0.0`)
- Gpufit/java/gpufit/src/main/java/com/github/gpufit/Gpufit.java (String VERSION = "1.0.0";)

Push to Github afterwards (you can add a Git tag for the new version with git tag -a vX.Y.Z -m "Version X.Y.Z release" followed by git push --tags).

## Convert Documentation from restructured text to html/latex

Use documentation_create_latex.bat in this folder or do it manually using sphinx and docs/make.bat.

## Use CMAKE to generate the project

- CUDA_ARCHITECTURE must be set to All (it is by default)
- CUDA toolkit 11.X is used for all builds (must be installed before)
- Build directory for Visual Studio 2019 Win64 is BUILD_BASE_PATH/VC16x64-11.X
- Matlab, Python, Java, Latex (e.g. Miktex) must be available

See also [Build from sources](https://gpufit.readthedocs.io/en/latest/installation.html#building-from-source-code) for instructions.

## Build for Win64

Everything should run through and the tests should execute successfully. Also run the Gpufit_Cpufit_Performance_Comparison.

- Configuration RelWithDebInfo is used for all builds!
- With Visual Studio 2019 Win64 build target PYTHON_WHEEL, MATLAB_GPUFIT_PACKAGE, JAVA_PACKAGE, Gpufit_Cpufit_Performance_Comparison, Tests and DOCUMENTATION_PDFLATEX (SOURCE_BASE_PATH\docs\_build\latex\Gpufit.pdf will be created from Gpufit.tex at the same location)

## Run the examples for the Bindings

In Matlab, Python and Java.

## Call the assemble script

create_package.bat %1 %2 %3

with 

- %1 is the BUILD_BASE_PATH (the path containing the various (see below) CMake generated Visual Studio projects)

- %2 is the VERSION (e.g. 1.0.0)

- %3 is the SOURCE_BASE_PATH (the path containing the sources)

The output is a folder (BUILD_BASE_PATH/Gpufit-VERSION) which is also zipped if 7-Zip is available.

## Retrieve the hash for the current commit in GIT

git rev-parse --verify --short HEAD

and add as suffix to build file name (this specifies the source commit state)