Example application for the Gpufit library (https://github.com/gpufit/Gpufit)
which implements Levenberg Marquardt curve fitting in CUDA.

Requirements
------------

- A CUDA capable graphics card with a recent Nvidia graphics driver
  (at least 367.48 / July 2016)
- Windows
- >1.5 GB of free RAM

Running
-------

Start "Gpufit_Cpufit_Performance_Comparison.exe" to see a speed comparison of
GPU and CPU implementation.

Output
------

The accurate execution of the performance comparison example shows the version
number of the installed CUDA driver and the CUDA runtime Gpufit was built with.

EXAMPLE:
  CUDA runtime version: 8.0
  CUDA driver version:  9.0

In the next step the successful generation of test data is indicated by three
full progress bars.

EXAMPLE:

                                -------------------------
  Generating test parameters    |||||||||||||||||||||||||
                                -------------------------
                                -------------------------
  Generating data               |||||||||||||||||||||||||
                                -------------------------
                                -------------------------
  Adding noise                  |||||||||||||||||||||||||
                                -------------------------
								
The results of the performance comparison between Gpufit and Cpufit are shown
in a table. The results demonstrate the performance benefit of Gpufit compared
to Cpufit executing the fitting process for various number of fits in a range
of 10 - 10000000. The execution speed is expressed in fits per second. If the
execution time was not measurable, the speed is expressed as infinite.

EXAMPLE:

    Number  | Cpufit speed  | Gpufit speed  | Performance
   of fits  |     (fits/s)  |     (fits/s)  | gain factor
  -------------------------------------------------------
        10  |          inf  |           92  |        0.00
       100  |          inf  |         6667  |        0.00
      1000  |        66667  |          inf  |         inf
     10000  |        58480  |       666667  |       11.40
    100000  |        59916  |      2173913  |       36.28
   1000000  |        59898  |      2469136  |       41.22
  10000000  |        60957  |      3038590  |       49.85

Troubleshooting
---------------

MESSAGE:

  CUDA runtime version: 0.0
  CUDA driver version:  7.5

  The CUDA runtime version is not compatible with the current graphics driver.
  Please update the driver, or re-build Gpufit from source using a compatible
  version of the CUDA toolkit.
  
  Skipping Gpufit computations.
  
BEHAVIOR:

  The example executes Cpufit skipping Gpufit. Only computation speed of Cpufit
  is shown in the results table.
  
SOLUTION:

  A common reason for this error message is an outdated Nvidia graphics driver.
  In most cases updating the graphics card driver will solve this error. For
  older graphics cards which are not supported by the CUDA toolkit used for
  building Gpufit, re-compile Gpufit using an earlier version of the CUDA
  toolkit which is compatible with the graphics driver.

MESSAGE:
  
  CUDA runtime version: 0.0
  CUDA driver version:  0.0
  
  No CUDA enabled graphics card detected.
  
  Skipping Gpufit computations.

BEHAVIOR:

  The example executes Cpufit skipping Gpufit. Only computation speed of Cpufit
  is shown in the results table.
  
SOLUTION:

  The execution of Gpufit requires a CUDA enabled graphics card.
  Ensure, that the host PC has installed a CUDA enabled graphics card.