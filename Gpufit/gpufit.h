#ifndef GPU_FIT_H_INCLUDED
#define GPU_FIT_H_INCLUDED

#ifdef __linux__
#define VISIBLE __attribute__((visibility("default")))
#endif

#ifdef _WIN32
#define VISIBLE
#endif

#include <cstddef>
#include <stdexcept>
#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

VISIBLE int gpufit
(
    size_t n_fits,
    size_t n_points,
    double * data,
    double * weights,
    int model_id,
    double * initial_parameters,
    double tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    size_t user_info_size,
    char * user_info,
    double * output_parameters,
    int * output_states,
    double * output_chi_squares,
    int * output_n_iterations,
    double * lambda_info
) ;

VISIBLE char const * gpufit_get_last_error() ;

// returns 1 if cuda is available and 0 otherwise
VISIBLE int gpufit_cuda_available();

VISIBLE int gpufit_get_cuda_version(int * runtime_version, int * driver_version);

VISIBLE int gpufit_portable_interface(int argc, void *argv[]);

#ifdef __cplusplus
}
#endif

#endif // GPU_FIT_H_INCLUDED
