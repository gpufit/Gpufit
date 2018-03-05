#ifndef GPUFIT_CUDA_KERNELS_CUH_INCLUDED
#define GPUFIT_CUDA_KERNELS_CUH_INCLUDED

#include <device_launch_parameters.h>

void configure_model(ModelID const model_id, int & n_parameters, int & n_dimensions);

extern __global__ void convert_pointer(
    double ** pointer_to_pointer,
    double * pointer,
    int const n_pointers,
    int const size,
    int const * skip);

extern __global__ void cuda_calculate_interim_euclidian_norms(
    double * norms,
    double const * vectors,
    int const n_points,
    int const n_fits,
    int const n_parameters,
    int const n_parameters_to_fit,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_complete_euclidian_norms(
    double * norms,
    int const n_blocks_per_fit,
    int const n_fits,
    int const n_parameters,
    int const * finished);

extern __global__ void cuda_sum_chi_square_subtotals(
    double * chi_squares,
    int const n_blocks_per_fit,
    int const n_fits,
    int const * finished);

extern __global__ void cuda_check_fit_improvement(
    int * iteration_failed,
    double const * chi_squares,
    double const * prev_chi_squares,
    int const n_fits,
    int const * finished);

extern __global__ void cuda_calculate_chi_squares(
    double * chi_squares,
    int * states,
    double const * data,
    double const * values,
    double const * weights,
    int const n_points,
    int const n_fits,
    int const estimator_id,
    int const * finished,
    int const n_fits_per_block,
    char * user_info,
    std::size_t const user_info_size);

extern __global__ void cuda_sum_gradient_subtotals(
    double * gradients,
    int const n_blocks_per_fit,
    int const n_fits,
    int const n_parameters,
    int const * skip,
    int const * finished);

extern __global__ void cuda_calculate_gradients(
    double * gradients,
    double const * data,
    double const * values,
    double const * derivatives,
    double const * weights,
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
    double * hessians,
    double const * data,
    double const * values,
    double const * derivatives,
    double const * weights,
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
    double * scaling_vectors,
    double const * hessians,
    int const n_parameters,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_init_scaled_hessians(
    double * scaled_hessians,
    double const * hessians,
    int const n_fits,
    int const n_parameters,
    int const * finished,
    int const * lambda_accepted,
    int const * newton_step_accepted);

extern __global__ void cuda_modify_step_widths(
    double * hessians,
    double const * lambdas,
    double * scaling_vectors,
    unsigned int const n_parameters,
    int const * finished,
    int const n_fits_per_block,
    int const * lambda_accepted,
    int const * newton_step_accepted);

extern __global__ void cuda_calc_curve_values(
    double const * parameters,
    int const n_fits,
    int const n_points,
    int const n_parameters,
    int const * finished,
    double * values,
    double * derivatives,
    int const n_fits_per_block,
    int const n_blocks_per_fit,
    ModelID const model_id,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size);

extern __global__ void cuda_update_parameters(
    double * parameters,
    double * prev_parameters,
    double const * deltas,
    int const n_parameters_to_fit,
    int const * parameters_to_fit_indices,
    int const * finished,
    int const n_fits_per_block);

extern __global__ void cuda_check_for_convergence(
    int * finished,
    double const tolerance,
    int * states,
    double const * chi_squares,
    double const * prev_chi_squares,
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
    double * lambdas,
    double * chi_squares,
    double * prev_chi_squares,
    double * function_parameters,
    double const * prev_parameters,
    int const n_fits,
    int const n_parameters);

__global__ void cuda_update_temp_derivatives(
    double * temp_derivatives,
    double const * derivatives,
    int const * iteration_failed,
    int const n_fits_per_block,
    int const n_blocks_per_fit,
    int const * finished,
    int const n_parameters,
    int const n_points);

__global__ void cuda_multiply(
    double * products,
    double const * multiplicands,
    double const * multipliers,
    int const * skip,
    int const n_vectors,
    int const vector_size,
    int const * skip_2,
    int const * not_skip_3);

__global__ void cuda_multiply_matrix_vector(
    double * products,
    double const * matrices,
    double const * vectors,
    int const n_rows,
    int const n_cols,
    int const n_fits_per_block,
    int const n_blocks_per_fit,
    int const * skip);

__global__ void cuda_initialize_step_bounds(
    double * step_bounds,
    double * scaled_parameters,
    int const * finished,
    int const n_fits,
    int const n_parameters);

__global__ void cuda_adapt_step_bounds(
    double * step_bounds,
    double const * scaled_delta_norms,
    int const * finished,
    int const n_fits);

__global__ void cuda_update_step_bounds(
    double * step_bounds,
    double * lambdas,
    double const * approximation_ratios,
    double const * actual_reductions,
    double const * directive_derivatives,
    double const * chi_squares,
    double const * prev_chi_squares,
    double const * scaled_delta_norms,
    int const * finished,
    int const n_fits);

__global__ void cuda_calc_phis(
    double * phis,
    double * phi_derivatives,
    double * inverted_hessians,
    double * scaled_deltas,
    double * scaled_delta_norms,
    double * temp_vectors,
    double const * scaling_vectors,
    double const * step_bounds,
    int const n_parameters,
    int const * finished,
    int const * lambda_accepted,
    int const * newton_step_accepted,
    int const n_fits_per_block);

__global__ void cuda_adapt_phi_derivatives(
    double * phi_derivatives,
    double const * step_bounds,
    double const * scaled_delta_norms,
    int const * finished);

__global__ void cuda_check_phi(
    int * newton_step_accepted,
    double const * phis,
    double const * step_bounds,
    int const * finished,
    int const n_fits);

__global__ void cuda_check_abs_phi(
    int * lambda_accepted,
    int const * newton_step_accepted,
    double const * phis,
    double const * step_bounds,
    int const * finished,
    int const n_fits);

__global__ void cuda_init_lambda_bounds(
    double * lambdas,
    double * lambda_lower_bounds,
    double * lambda_upper_bounds,
    double * scaled_gradients,
    double const * scaled_delta_norms,
    double const * phis,
    double const * phi_derivatives,
    double const * step_bounds,
    double const * gradients,
    double const * scaling_vectors,
    int const * finished,
    int const n_fits,
    int const n_parameters,
    int const * newton_step_accepted);

__global__ void cuda_update_lambdas(
    double * lambdas,
    double * lambda_lower_bounds,
    double * lambda_upper_bounds,
    double const * phis,
    double const * phi_derivatives,
    double const * step_bounds,
    int const * finished,
    int const * lambda_accepted,
    int const * newton_step_accepted,
    int const n_fits);

__global__ void cuda_calc_approximation_quality(
    double * predicted_reductions,
    double * actual_reductions,
    double * directive_derivatives,
    double * approximation_ratios,
    double * derivatives_deltas,
    double const * scaled_delta_norms,
    double const * chi_squares,
    double const * prev_chi_squares,
    double const * lambdas,
    int const * finished,
    int const n_fits,
    int const n_points);

#endif
