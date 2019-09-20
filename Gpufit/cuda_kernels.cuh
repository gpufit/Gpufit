#ifndef GPUFIT_CUDA_KERNELS_CUH_INCLUDED
#define GPUFIT_CUDA_KERNELS_CUH_INCLUDED

#include <device_launch_parameters.h>
#include "definitions.h"

void configure_model(ModelID const model_id, int & n_parameters, int & n_dimensions);

extern __global__ void cuda_sum_chi_square_subtotals(
    REAL * chi_squares,
    REAL const * subtotals,
    int const n_blocks_per_fit,
    int const n_fits,
    int const * finished);

extern __global__ void cuda_check_fit_improvement(
    int * iteration_failed,
    REAL const * chi_squares,
    REAL const * prev_chi_squares,
    int const n_fits,
    int const * finished);

extern __global__ void cuda_calculate_chi_squares(
    REAL * chi_squares,
    int * states,
    REAL const * data,
    REAL const * values,
    REAL const * weights,
    int const n_points,
    int const n_fits,
    int const estimator_id,
    int const * finished,
    int const n_fits_per_block,
    char * user_info,
    std::size_t const user_info_size);

extern __global__ void cuda_sum_gradient_subtotals(
    REAL * gradients,
    REAL const * subtotals,
    int const n_blocks_per_fit,
    int const n_fits,
    int const n_parameters,
    int const * skip,
    int const * finished);

extern __global__ void cuda_calculate_gradients(
    REAL * gradients,
    REAL const * data,
    REAL const * values,
    REAL const * derivatives,
    REAL const * weights,
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
    REAL * hessians,
    REAL const * data,
    REAL const * values,
    REAL const * derivatives,
    REAL const * weights,
    int const n_fits,
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
    REAL * hessians,
    REAL const * lambdas,
    REAL * scaling_vectors,
    unsigned int const n_parameters,
    int const * iteration_failed,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_calc_curve_values(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    int const n_parameters,
    int const * finished,
    REAL * values,
    REAL * derivatives,
    int const n_fits_per_block,
    int const n_blocks_per_fit,
    ModelID const model_id,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size);

extern __global__ void cuda_update_parameters(
    REAL * parameters,
    REAL * prev_parameters,
    REAL const * parameter_constraints,
    REAL const * deltas,
    int const n_parameters_to_fit,
    int const * parameters_to_fit_indices,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_check_for_convergence(
    int * finished,
    REAL const tolerance,
    int * states,
    REAL const * chi_squares,
    REAL const * prev_chi_squares,
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
    REAL * lambdas,
    REAL * chi_squares,
    REAL * prev_chi_squares,
    REAL * function_parameters,
    REAL const * prev_parameters,
    int const n_fits,
    int const n_parameters);

extern __global__ void cuda_update_state_after_solving(
    int const n_fits,
    int const * singular_checks,
    int const * finished,
    int * states);

#endif
