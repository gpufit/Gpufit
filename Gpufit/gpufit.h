#ifndef GPU_FIT_H_INCLUDED
#define GPU_FIT_H_INCLUDED

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
#define STATE_GPU_NOT_READY 4

// gpufit return state
#define STATUS_OK 0
#define STATUS_ERROR -1

#ifdef __cplusplus
extern "C" {
#endif


#include <cstddef>


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
