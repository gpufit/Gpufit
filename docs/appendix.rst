========
Appendix
========

Levenberg-Marquardt algorithm
-----------------------------

A flowchart of the implementation of the Levenberg-Marquardt algorithm is given in :numref:`appendix-gpufit-flowchart`.

.. _appendix-gpufit-flowchart:

.. figure:: /images/gpufit_program_flow_skeleton_v3.png
   :width: 14 cm
   :align: center

   Levenberg-Marquardt algorithm flow as implemented in the Gpufit library.
   
   
Performance comparison to other GPU benchmarks
----------------------------------------------

Using the bundled application (initial release created with CUDA runtime 8.0) to estimate the fitting speed per second of 10 million fits for various CUDA capable
graphics cards of various architectures (on different computers with different versions of graphics drivers) we can
compare to the results of Passmark G3D. By and large, the results seem to correlate, i.e. a high Passmark G3D score
also relates to a high Gpufit fitting speed.

.. figure:: /images/GPUfit_PassmarkG3D_relative_performance.png
   :width: 14 cm
   :align: center

   Performance of Gpufit vs Passmark G3D