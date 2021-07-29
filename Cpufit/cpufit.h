#ifndef CPU_FIT_H_INCLUDED
#define CPU_FIT_H_INCLUDED

#ifdef __linux__
#define VISIBLE __attribute__((visibility("default")))
#endif

#ifdef _WIN32
#define VISIBLE
#endif

#include <cstddef>
#include "../Gpufit/constants.h"
#include "../Gpufit/definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

VISIBLE int cpufit
(
    std::size_t n_fits,
    std::size_t n_points,
    REAL * data,
    REAL * weights,
    int model_id,
    REAL * initial_parameters,
    REAL tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    std::size_t user_info_size,
    char * user_info,
    REAL * output_parameters,
    int * output_states,
    REAL * output_chi_squares,
    int * output_n_iterations
) ;

VISIBLE char const * cpufit_get_last_error() ;

#ifdef __cplusplus
}
#endif

#endif // CPU_FIT_H_INCLUDED
