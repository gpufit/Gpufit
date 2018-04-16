#ifndef GPUFIT_CUDA_GAUSS_JORDAN_CUH_INCLUDED
#define GPUFIT_CUDA_GAUSS_JORDAN_CUH_INCLUDED

#include <device_launch_parameters.h>
#include "definitions.h"

extern __global__ void cuda_gaussjordan(
    REAL * delta,
    REAL const * beta,
    REAL const * alpha,
    int const * skip_calculation,
    int * singular,
    std::size_t const n_equations,
    std::size_t const n_equations_pow2);

#endif