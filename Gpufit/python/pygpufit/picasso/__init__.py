"""
picasso.fitting
~~~~~~~~~~~~~~~

Levenberg-Marquardt PSF fitters, on the CPU and on CUDA GPUs.

Every module here is a port of Gpufit (https://github.com/gpufit/Gpufit)
- its LM driver, its damping rule, its least-squares and Poisson
maximum-likelihood estimators, and its Gaussian and
cubic-spline PSF models - written in Python and compiled by Numba, for the CPU
(``numba.jit``) and for CUDA GPUs (``numba.cuda.jit``).

===================  =========================================================
module               contents
===================  =========================================================
:mod:`splinefit`     CPU cubic-spline PSF fitting: the LM driver, the four
                     spline models and both estimators. The reference
                     implementation the CUDA kernels are transcribed from, and
                     the owner of every constant that decides *where a fit
                     stops*.
:mod:`lmfit_cuda`    CUDA device machinery shared by the GPU backends: the
                     damping, the Gauss-Jordan solve, the estimators and the
                     host-side launch bookkeeping.
:mod:`splinefit_cuda`  GPU cubic-spline PSF fitting; interchangeable with
                     ``splinefit``.
:mod:`gaussfit`      CPU 2D Gaussian PSF fitting (spherical, elliptical and
                     rotated), over the same LM driver.
:mod:`gaussfit_cuda`   GPU 2D Gaussian PSF fitting (spherical, elliptical and
                     rotated).
:mod:`precision`     Uncertainties of the fitted parameters: the closed-form
                     precisions and the numerically inverted Fisher matrices
                     (Cramer-Rao bounds), on the CPU and on CUDA GPUs. Not
                     part of the Gpufit port.
===================  =========================================================

References
----------
Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
Licence (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""
