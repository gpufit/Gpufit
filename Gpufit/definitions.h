#ifndef GPUFIT_DEFINITIONS_H_INCLUDED

    #define GPUFIT_DEFINITIONS_H_INCLUDED

    // Precision
    #ifdef GPUFIT_DOUBLE
        #define REAL double
        #define DECOMPOSE_LUP cublasDgetrfBatched
        #define SOLVE_LUP cublasDgetrsBatched
    #else
        #define REAL float
        #define DECOMPOSE_LUP cublasSgetrfBatched
        #define SOLVE_LUP cublasSgetrsBatched
    #endif // GPUFIT_DOUBLE


    // Status
    #include <stdexcept>

    #define CUDA_CHECK_STATUS( cuda_function_call ) \
        if (cudaError_t const status = cuda_function_call) \
        { \
            throw std::runtime_error( cudaGetErrorString( status ) ) ; \
        }

    #if (defined(_WIN64) || defined(__x86_64__) || defined(__LP64__)) 
        #define ARCH_64
    #endif // (defined(_WIN64) || defined(__x86_64__) || defined(__LP64__))

    #if (defined(USE_CUBLAS) && !defined(ARCH_64))
        #undef USE_CUBLAS
    #endif // (defined(USE_CUBLAS) && !defined(ARCH_64))

    #if defined(USE_CUBLAS)

        #define SOLVE_EQUATION_SYSTEMS() solve_equation_systems_lup()

    #else // defined(USE_CUBLAS)

        #define cublasHandle_t int
        #define SOLVE_EQUATION_SYSTEMS() solve_equation_systems_gj()

    #endif // defined(USE_CUBLAS) 

#endif // GPUFIT_DEFINITIONS_H_INCLUDED
