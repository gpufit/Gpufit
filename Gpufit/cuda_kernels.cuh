#ifndef GPUFIT_CUDA_KERNELS_CUH_INCLUDED
#define GPUFIT_CUDA_KERNELS_CUH_INCLUDED

#include <device_launch_parameters.h>

void configure_model(ModelID const model_id, int & n_parameters, int & n_dimensions);

extern __global__ void cuda_sum_chi_square_subtotals(
    float * chi_squares,
    int const n_blocks_per_fit,
    int const n_fits,
    int const * finished);

extern __global__ void cuda_check_fit_improvement(
    int * iteration_failed,
    float const * chi_squares,
    float const * prev_chi_squares,
    int const n_fits,
    int const * finished);

extern __global__ void cuda_calculate_chi_squares(
    float * chi_squares,
    int * states,
    float const * data,
    float const * values,
    float const * weights,
    int const n_points,
    int const n_fits,
    int const estimator_id,
    int const * finished,
    int const n_fits_per_block,
    char * user_info,
    std::size_t const user_info_size);

extern __global__ void cuda_sum_gradient_subtotals(
    float * gradients,
    int const n_blocks_per_fit,
    int const n_fits,
    int const n_parameters,
    int const * skip,
    int const * finished);

extern __global__ void cuda_calculate_gradients(
    float * gradients,
    float const * data,
    float const * values,
    float const * derivatives,
    float const * weights,
    int const n_points,
    int const n_fits,
    int const n_parameters,
    int const n_parameters_to_fit,
    int const * parameters_to_fit_indices,
	int const estimator_id,
    int const * finished,
    int const * skip,
    int const n_fits_per_block,
    char * user_info,
    std::size_t const user_info_size);

extern __global__ void cuda_calculate_hessians(
    float * hessians,
    float const * data,
    float const * values,
    float const * derivatives,
    float const * weights,
    int const n_points,
    int const n_parameters,
    int const n_parameters_to_fit,
    int const * parameters_to_fit_indices,
	int const estimator_id,
    int const * skip,
    int const * finished,
    char * user_info,
    std::size_t const user_info_size);

extern __global__ void cuda_modify_step_widths(
    float * hessians,
    float const * lambdas,
    unsigned int const n_parameters,
    int const * iteration_failed,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_calc_curve_values(
    float const * parameters,
    int const n_fits,
    int const n_points,
    int const n_parameters,
    int const * finished,
    float * values,
    float * derivatives,
    int const n_fits_per_block,
    int const n_blocks_per_fit,
    ModelID const model_id,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size);

extern __global__ void cuda_update_parameters(
    float * parameters,
    float * prev_parameters,
    float const * deltas,
    int const n_parameters_to_fit,
    int const * parameters_to_fit_indices,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_check_for_convergence(
    int * finished,
    float const tolerance,
    int * states,
    float const * chi_squares,
    float const * prev_chi_squares,
    int const iteration,
    int const max_n_iterations,
    int const n_fits);

extern __global__ void cuda_evaluate_iteration(
    int * all_finished,
    int * n_iterations,
    int * finished,
    int const iteration,
    int const * states,
    int const n_fits);

extern __global__ void cuda_prepare_next_iteration(
    float * lambdas,
    float * chi_squares,
    float * prev_chi_squares,
    float * function_parameters,
    float const * prev_parameters,
    int const n_fits,
    int const n_parameters);

extern __global__ void cuda_update_state_after_gaussjordan(
    int const n_fits,
    int const * singular_checks,
    int * states);

#endif
