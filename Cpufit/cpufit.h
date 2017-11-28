#ifndef CPU_FIT_H_INCLUDED
#define CPU_FIT_H_INCLUDED

#include <cstddef>
#include "../Gpufit/constants.h"

#ifdef __cplusplus
extern "C" {
#endif

int cpufit
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

char const * cpufit_get_last_error() ;

#ifdef __cplusplus
}
#endif

#endif // CPU_FIT_H_INCLUDED
