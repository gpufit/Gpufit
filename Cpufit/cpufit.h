#ifndef CPU_FIT_H_INCLUDED
#define CPU_FIT_H_INCLUDED

// fitting model ID
#define GAUSS_1D 0
#define GAUSS_2D 1
#define GAUSS_2D_ELLIPTIC 2
#define GAUSS_2D_ROTATED 3
#define CAUCHY_2D_ELLIPTIC 4
#define LINEAR_1D 5

// estimator ID
#define LSE 0
#define MLE 1

// fit state
#define STATE_CONVERGED 0
#define STATE_MAX_ITERATION 1
#define STATE_SINGULAR_HESSIAN 2
#define STATE_NEG_CURVATURE_MLE 3

// cpufit return state
#define STATUS_OK 0
#define STATUS_ERROR -1

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
