.. _installation-and-testing:

========================
Installation and Testing
========================

The Gpufit library can be used in several ways. When using a pre-compiled
binary version of Gpufit, the Gpufit functions may be accessed directly via 
a dynamic linked library (e.g. Gpufit.dll) or via the external bindings to 
Gpufit (e.g. the Matlab or Python bindings). For more information on the
Gpufit interface, see :ref:`api-description`, or for details of the external
bindings see :ref:`external-bindings`.

This section describes how to compile Gpufit, including generating its 
external bindings, from source code. Building from source is necessary when
a fit model function is added or changed, or if a new fit estimator is required.
Building the library may also be useful for compiling the code using a 
specific version of the CUDA toolkit, or for a particular CUDA compute 
capability. 

Gpufit binary distribution
++++++++++++++++++++++++++

A binary distribution of the Gpufit library is available for **Windows**.
Use of this distribution requires only a CUDA-capable graphics card, and an
updated Nvidia graphics driver. The binary package contains:

- The Gpufit SDK, which consists of the 32-bit and 64-bit DLL files, and 
  the Gpufit header file which contains the function definitions. The Gpufit
  SDK is intended to be used when calling Gpufit from an external application
  written in e.g. C code.
- The performance test application, which serves to test that Gpufit is 
  correctly installed, and to check the performance of the CPU and GPU hardware.
- Matlab 32 bit and 64 bit bindings, with Matlab examples.
- Python version 2.x and version 3.x bindings (compiled as wheel files) and
  Python examples.
- This manual in PDF format.

To re-build the binary distribution, see the instructions located in 
package/README.md.

Building from source code
+++++++++++++++++++++++++

This section describes how to build Gpufit from source code. Note that as of
the initial release of Gpufit, the source code has been tested only with the 
Microsoft Visual Studio compiler.

Prerequisites
-------------

The following tools are required in order to build Gpufit from source.

*Required*

* CMake_ 3.7 or later
* A C/C++ Compiler

  * Linux: GCC 4.7
  * Windows: Visual Studio 2013 or 2015

* CUDA_ Toolkit 6.5 or later [#]_

.. [#] Note that it is recommended to use the newest available stable release of the CUDA Toolkit which is compatible
    with the compiler (e.g. Visual Studio 2015 is required in order to use CUDA Toolkit 8.0). Some older graphics cards
    may only be supported by CUDA Toolkit version 6.5 or earlier. Also, when using CUDA Toolkit version 6.5, please use
    the version with support for GTX9xx GPUs, available `here <https://developer.nvidia.com/cuda-downloads-geforce-gtx9xx>`__.

*Optional*

* Boost_ 1.58 or later (required if you want to build the tests)
* MATLAB_ if building the MATLAB bindings (minimum version Matlab 2012a)
* Python_ if building the Python bindings (Python version 2.x or 3.x)

Source code availability
------------------------

The source code is available in an open repository hosted at Github, at the 
following URL.

.. code-block:: bash

    https://github.com/gpufit/Gpufit.git

To obtain the code, Git may be used to clone the repository, or a current 
snapshot may be downloaded directly from Github as Gpufit-master.zip_.

Compiler configuration via CMake
--------------------------------

CMake is an open-source tool designed to build, test, and package software. 
It is used to control the software compilation process using compiler 
independent configuration files, and generate native makefiles and workspaces 
that can be used in the compiler environment. In this section we provide a
simple example of how to use CMake in order to generate the input files for the
compiler (e.g. the Visual Studio solution file), which can then be used to 
compile Gpufit.

First, identify the directory which contains the Gpufit source code 
(for example, on a Windows computer the Gpufit source code may be stored in 
*C:\\Sources\\Gpufit*). Next, create a build directory outside the
source code source directory (e.g. *C:\\Sources\\Gpufit-build-64*). Finally, 
run cmake to configure and generate the compiler input files. The following
commands, executed from the command prompt, assume that the cmake executable
(e.g. *C:\\Program Files\\CMake\\bin\\cmake.exe*) is automatically found 
via the PATH environment variable (if not, the full path to cmake.exe must be
specified). This example also assumes that the source and build directories
have been set up as specified above.

.. code-block:: bash

    cd C:\Sources\Gpufit-build-64
    cmake -G "Visual Studio 12 2013 Win64" C:\Sources\Gpufit

Note that in this example the *-G* flag has been used to specify the 
64-bit version of the Visual Studio 12 compiler. This flag should be changed
depending on the compiler used, and the desired architecture 
(e.g. 32- or 64-bit). Further details of the CMake command line arguments
can be found `here <https://cmake.org/cmake/help/latest/manual/cmake.1.html>`__.

There is also a graphical user interface available for CMake, which simplifies
the configuration and generation steps. For further details, see
`Running CMake <https://cmake.org/runningcmake/>`_.

Common issues encountered during CMake configuration
----------------------------------------------------

**Boost NOT found - skipping tests!**

If you want to build the tests and Boost is not found automatically, set the 
CMake variable BOOST_ROOT to the corresponding directory, and configure again.

**Specify CUDA_ARCHITECTURES set**

If you need a specific CUDA architecture, set CUDA_ARCHITECTURES according 
to CUDA_SELECT_NVCC_ARCH_FLAGS_.

**CMake finds last installed CUDA toolkit version by default**

If there are multiple CUDA toolkits installed on the computer, CMake 3.7.1 
seems to find by default the lowest installed version. In this case set the desired CUDA
version manually (e.g. by editing the CUDA_TOOLKIT_ROOT_DIR variable in CMake).

**Specify CUDA version to use**

Set CUDA_BIN_PATH before running CMake or CUDA_TOOLKIT_ROOT_DIR after 
first CMAKE configuration to the installation folder of the desired 
CUDA version.

**Required CUDA version**

When using Microsoft Visual Studio 2015, the minimum required CUDA Toolkit 
version is 8.0.

**Python launcher**

Set Python_WORKING_DIRECTORY to a valid directory, it will be added to the 
Python path.

**Matlab launcher**

Set Matlab_WORKING_DIRECTORY to a valid directory, it will be added to 
the Matlab path.

Compiling Gpufit on Windows
---------------------------

After configuring and generating the solution files using CMake, go to the 
desired build directory and open Gpufit.sln using Visual Studio. Select the
"Debug" or "Release" build options, as appropriate. Select the build target
"ALL_BUILD", and build this target. If the build process completes
without errors, the Gpufit binary files will be created in the corresponding 
"Debug" or "Release" folders in the build directory.

The unit tests can be executed by building the target "RUN_TESTS" or by 
starting the created executables in the output directory from
the command line.

Linux
-----

Gpufit has not yet been officially tested on a computer running a Linux variant 
with a CUDA capable graphics card. However, satisfying the Prerequisites_ and
using CMake, we estimate that the library should build in principle and one
should also be able to run the examples on Linux.

MacOS
-----

Gpufit has not yet been officially tested on a computer running MacOS with a 
CUDA capable graphics card. However, satisfying the Prerequisites_ and using
CMake, we estimate that the library should build in principle and one
should also be able to run the examples on MacOS.

Running the performance test
++++++++++++++++++++++++++++

The Gpufit performance test is a program which verifies the correct function
of Gpufit, and tests the fitting speed in comparison with the same algorithm
executed on the CPU.

If Gpufit was built from source, running the build target 
GPUFIT_CPUFIT_Performance_Comparison will run the test, which executes the 
fitting process multiple times, varying the number of fits per function call.
The execution time is measured in each case and the relative speed improvement 
between the GPU and the CPU is calculated. A successful run of the performance
test also indicates also that Gpufit is functioning correctly.

The performance comparison is also included in the Gpufit binary distribution
as a console application. An example of the program's output is
shown in :numref:`installation-gpufit-cpufit-performance-comparison`.

.. _installation-gpufit-cpufit-performance-comparison:

.. figure:: /images/GPUFIT_CPUFIT_Performance_Comparison.png
   :width: 10 cm
   :align: center

   Output of the Gpufit vs Cpufit performance comparison

