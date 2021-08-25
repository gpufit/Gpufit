.. highlight:: text

============
Introduction
============

Gpufit is a GPU-accelerated CUDA implementation of the Levenberg-Marquardt 
algorithm. It was developed to meet the need for a high performance, general-
purpose nonlinear curve fitting software library which is publicly available
and open source.

Optimization algorithms are ubiquitous tools employed in many field of science 
and technology. One such algorithm for numerical, non-linear optimization is the 
Levenberg-Marquardt algorithm (LMA). The LMA combines elements of the method of 
steepest descent and Newton's method, and has become a standard algorithm for 
least-squares fitting. Box constraints on parameter values can be added as suitable
projections during the optimization steps.

Although the LMA is, in itself, an efficient optimization algorithm, 
applications requiring many iterations of this procedure may encounter 
limitations due to the sheer number of calculations involved. The time required 
for the convergence of a fit, or a set of fits, can determine an application's 
feasibility, e.g. in the context of real-time data processing and feedback 
systems. Alternatively, in the case of very large datasets, the time required 
to solve a particular optimization problem may prove impractical.

In recent years, advanced graphics processing units (GPUs) and the development 
of general purpose GPU programming have enabled fast and parallelized computing 
by shifting calculations from the CPU to the GPU. The large number of 
independent computing units available on a modern GPU enables the rapid 
execution of many instructions in parallel, with an overall computation power 
far exceeding that of a CPU. Languages such as CUDA C and OpenCL allow GPU-
based programs to be developed in a manner similar to conventional software, but 
with an inherently parallelized structure. These developments have led to the 
creation of new GPU-accelerated tools, such as the Gpufit.

Gpufit supports cubic spline functions that can be used to approximate arbitrary (smooth) fit model functions.
In order to use them a spline representation of the model function must be provided (as an array of suitable spline
coefficients). See `Gpuspline on Github`_ for details on how to compute these spline representations.

This manual describes how to install and build the Gpufit library and its 
external bindings. Furthermore it details how to extend Gpufit by adding 
custom model functions as well as custom fit estimator functions.

The documentation includes:

- Instructions for building and installing Gpufit
- A detailed description of the C interface
- A description of the built-in model functions
- A description of the built-in goodness-of-fit estimator functions
- A detailed description of the external bindings to Matlab and Python
- Usage examples for C, Matlab, and Python
- Instructions for adding custom model functions or custom estimator functions

The current version of the Gpufit library is |GF_version| 
(`see homepage <http://github.com/gpufit/Gpufit>`_). This manual was compiled 
on |today|.

How to cite Gpufit
------------------

Gpufit was created by Mark Bates, Adrian Przybylski, Björn Thiel, and Jan Keller-Findeisen at the Max Planck Institute for Biophysical Chemistry, in Göttingen, Germany.

The development and maintenance of open-source software projects, such as Gpufit, requires significant time and resources from the project team.  If you use Gpufit in your research, **please cite our publication**.  A paper describing the Gpufit software was published in the journal Scientific Reports, and is available from the Scientific Reports website (open-access), [here](https://www.nature.com/articles/s41598-017-15313-9).

The citation for the Gpufit paper is as follows::

    Gpufit: An open-source toolkit for GPU-accelerated curve fitting  
    Adrian Przybylski, Björn Thiel, Jan Keller-Findeisen, Bernd Stock, and Mark Bates  
    Scientific Reports, vol. 7, 15722 (2017); doi: https://doi.org/10.1038/s41598-017-15313-9 

Hardware requirements
---------------------

Because the fit algorithm is implemented in CUDA C, a CUDA_-compatible graphics
card is required to run Gpufit. The minimum supported compute capability is 
2.0. More advanced GPU hardware will result in higher fitting performance.

Software requirements
---------------------

In addition to a compatible GPU, the graphics card driver installed on the 
host computer must be compatible with the version of the CUDA toolkit which 
was used to compile Gpufit. This may present an issue for older graphics 
cards or for computers running outdated graphics drivers.

At the time of its initial release in 2017, Gpufit was compiled with CUDA toolkit
version 8.0. Therefore, the Nvidia graphics driver installed on the host PC 
must be at least version 367.48 (released July 2016) in order to be compatible
with the binary files generated in this build.

When compatibility issues arise, there are two possible solutions. The best 
option is to update the graphics driver to a version which is compatible with
the CUDA toolkit used to build Gpufit. The second option is to re-compile 
Gpufit from source code, using an earlier version of the CUDA toolkit which is 
compatible with the graphics driver in question. However, this solution is 
likely to result in slower performance of the Gpufit code, since older versions 
of the CUDA toolkit are not as efficient.

Note that all CUDA-supported graphics cards should be compatible with
CUDA toolkit version 6.5. This is the last version of CUDA which supported 
GPUs with compute capability 1.x. In other words, an updated Nvidia graphics
driver should be available for all CUDA-enabled GPUs which is compatible with
toolkit version 6.5. 

If you are unsure if your graphics card is CUDA-compatible, a lists of CUDA
supported GPUs can be found `here <http://developer.nvidia.com/cuda-gpus>`_.
