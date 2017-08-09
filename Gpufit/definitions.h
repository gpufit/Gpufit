#ifndef GPUFIT_DEFINITIONS_H_INCLUDED
#define GPUFIT_DEFINITIONS_H_INCLUDED

    // Status
#include <stdexcept>
#define CUDA_CHECK_STATUS( cuda_function_call ) \
    if (cudaError_t const status = cuda_function_call) \
    { \
        throw std::runtime_error( cudaGetErrorString( status ) ) ; \
    }

#endif
