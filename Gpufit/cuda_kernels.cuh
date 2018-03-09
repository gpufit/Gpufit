#ifndef GPUFIT_CUDA_KERNELS_CUH_INCLUDED
#define GPUFIT_CUDA_KERNELS_CUH_INCLUDED

#include <device_launch_parameters.h>

void configure_model(ModelID const model_id, int & n_parameters, int & n_dimensions);

extern __global__ void convert_pointer(
    float ** pointer_to_pointer,
    float * pointer,
    int const n_pointers,
    int const size,
    int const * skip);

extern __global__ void cuda_calculate_interim_euclidian_norms(
    float * norms,
    float const * vectors,
    int const n_points,
    int const n_fits,
    int const n_parameters,
    int const n_parameters_to_fit,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_complete_euclidian_norms(
    float * norms,
    int const n_blocks_per_fit,
    int const n_fits,
    int const n_parameters,
    int const * finished);

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

__global__ void cuda_calc_scaling_vectors(
    float * scaling_vectors,
    float const * hessians,
    int const n_parameters,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_init_scaled_hessians(
    float * scaled_hessians,
    float const * hessians,
    int const n_fits,
    int const n_parameters,
    int const * finished,
    int const * lambda_accepted,
    int const * newton_step_accepted);

extern __global__ void cuda_modify_step_widths(
    float * hessians,
    float const * lambdas,
    float * scaling_vectors,
    unsigned int const n_parameters,
    int const * finished,
    int const n_fits_per_block,
    int const * lambda_accepted,
    int const * newton_step_accepted);

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

extern __global__ void cuda_check_all_lambdas(
    int * all_lambdas_accepted,
    int const * finished,
    int const * lambda_accepted,
    int const * newton_step_accepted,
    int const n_fits);

extern __global__ void cuda_prepare_next_iteration(
    float * lambdas,
    float * chi_squares,
    float * prev_chi_squares,
    float * function_parameters,
    float const * prev_parameters,
    int const n_fits,
    int const n_parameters);

__global__ void cuda_update_temp_derivatives(
    float * temp_derivatives,
    float const * derivatives,
    int const * iteration_failed,
    int const n_fits_per_block,
    int const n_blocks_per_fit,
    int const * finished,
    int const n_parameters,
    int const n_points);

__global__ void cuda_multiply(
    float * products,
    float const * multiplicands,
    float const * multipliers,
    int const * skip,
    int const n_vectors,
    int const vector_size,
    int const * skip_2,
    int const * not_skip_3);

__global__ void cuda_multiply_matrix_vector(
    float * products,
    float const * matrices,
    float const * vectors,
    int const n_rows,
    int const n_cols,
    int const n_fits_per_block,
    int const n_blocks_per_fit,
    int const * skip);

__global__ void cuda_initialize_step_bounds(
    float * step_bounds,
    float * scaled_parameters,
    int const * finished,
    int const n_fits,
    int const n_parameters);

__global__ void cuda_adapt_step_bounds(
    float * step_bounds,
    float const * scaled_delta_norms,
    int const * finished,
    int const n_fits);

__global__ void cuda_update_step_bounds(
    float * step_bounds,
    float * lambdas,
    float const * approximation_ratios,
    float const * actual_reductions,
    float const * directive_derivatives,
    float const * chi_squares,
    float const * prev_chi_squares,
    float const * scaled_delta_norms,
    int const * finished,
    int const n_fits);

__global__ void cuda_calc_phis(
    float * phis,
    float * phi_derivatives,
    float * inverted_hessians,
    float * scaled_deltas,
    float * scaled_delta_norms,
    float * temp_vectors,
    float const * scaling_vectors,
    float const * step_bounds,
    int const n_parameters,
    int const * finished,
    int const * lambda_accepted,
    int const * newton_step_accepted,
    int const n_fits_per_block);

__global__ void cuda_adapt_phi_derivatives(
    float * phi_derivatives,
    float const * step_bounds,
    float const * scaled_delta_norms,
    int const * finished,
    int const n_fits);

__global__ void cuda_check_phi(
    int * newton_step_accepted,
    float const * phis,
    float const * step_bounds,
    int const * finished,
    int const n_fits);

__global__ void cuda_check_abs_phi(
    int * lambda_accepted,
    int const * newton_step_accepted,
    float const * phis,
    float const * step_bounds,
    int const * finished,
    int const n_fits);

__global__ void cuda_init_lambda_bounds(
    float * lambdas,
    float * lambda_lower_bounds,
    float * lambda_upper_bounds,
    float * scaled_gradients,
    float const * scaled_delta_norms,
    float const * phis,
    float const * phi_derivatives,
    float const * step_bounds,
    float const * gradients,
    float const * scaling_vectors,
    int const * finished,
    int const n_fits,
    int const n_parameters,
    int const * newton_step_accepted);

__global__ void cuda_update_lambdas(
    float * lambdas,
    float * lambda_lower_bounds,
    float * lambda_upper_bounds,
    float const * phis,
    float const * phi_derivatives,
    float const * step_bounds,
    int const * finished,
    int const * lambda_accepted,
    int const * newton_step_accepted,
    int const n_fits);

__global__ void cuda_calc_approximation_quality(
    float * predicted_reductions,
    float * actual_reductions,
    float * directive_derivatives,
    float * approximation_ratios,
    float * derivatives_deltas,
    float const * scaled_delta_norms,
    float const * chi_squares,
    float const * prev_chi_squares,
    float const * lambdas,
    int const * finished,
    int const n_fits,
    int const n_points);

#endif
