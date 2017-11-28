#ifndef GPU_FIT_H_INCLUDED
#define GPU_FIT_H_INCLUDED

#include <cstddef>
#include <stdexcept>
#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

int gpufit
(
    size_t n_fits,
    size_t n_points,
    float * data,
    float * weights,
    int model_id,
    float * initial_parameters,
    float tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    size_t user_info_size,
    char * user_info,
    float * output_parameters,
    int * output_states,
    float * output_chi_squares,
    int * output_n_iterations
) ;

char const * gpufit_get_last_error() ;

// returns 1 if cuda is available and 0 otherwise
int gpufit_cuda_available();

int gpufit_get_cuda_version(int * runtime_version, int * driver_version);

int gpufit_portable_interface(int argc, void *argv[]);

#ifdef __cplusplus
}
#endif

#endif // GPU_FIT_H_INCLUDED
