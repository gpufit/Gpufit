#include "lm_fit.h"
#include <algorithm>
#include "cuda_kernels.cuh"

#ifdef USE_CUBLAS

void LMFitCUDA::solve_equation_systems_lup()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    // initialize components of equation systems
    gpu_data_.copy(gpu_data_.decomposed_hessians_, gpu_data_.hessians_, n_fits_ * info_.n_parameters_to_fit_ * info_.n_parameters_to_fit_);

    // decompose hessians
    cublasStatus_t lu_status_decopmposition = DECOMPOSE_LUP(
        gpu_data_.cublas_handle_,
        info_.n_parameters_to_fit_,
        gpu_data_.pointer_decomposed_hessians_,
        info_.n_parameters_to_fit_,
        gpu_data_.pivot_vectors_,
        gpu_data_.solution_info_,
        n_fits_);

    // initialize deltas with values of gradients
    gpu_data_.copy(gpu_data_.deltas_, gpu_data_.gradients_, n_fits_ * info_.n_parameters_to_fit_);

    // TODO: check solution_info
    int solution_info;

    // solve equation systems
    cublasStatus_t lu_status_solution
        = SOLVE_LUP(
        gpu_data_.cublas_handle_,
        CUBLAS_OP_N,
        info_.n_parameters_to_fit_,
        1,
        (REAL const **)(gpu_data_.pointer_decomposed_hessians_.data()),
        info_.n_parameters_to_fit_,
        gpu_data_.pivot_vectors_,
        gpu_data_.pointer_deltas_,
        info_.n_parameters_to_fit_,
        &solution_info,
        n_fits_);
}

#else //USE_CUBLAS

void LMFitCUDA::solve_equation_systems_gj()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    int n_parameters_pow2 = 1;

    while (n_parameters_pow2 < info_.n_parameters_to_fit_)
    {
        n_parameters_pow2 *= 2;
    }

    //set up to run the Gauss Jordan elimination
    int const n_equations = info_.n_parameters_to_fit_;
    int const n_solutions = n_fits_;

    threads.x = n_equations + 1;
    threads.y = n_equations;
    blocks.x = n_solutions;

    //set the size of the shared memory area for each block
    int const shared_size
        = sizeof(REAL) * ((threads.x * threads.y)
            + n_parameters_pow2 + n_parameters_pow2);

    //run the Gauss Jordan elimination
    cuda_gaussjordan <<< blocks, threads, shared_size >>>(
        gpu_data_.deltas_,
        gpu_data_.gradients_,
        gpu_data_.hessians_,
        gpu_data_.finished_,
        gpu_data_.solution_info_,
        info_.n_parameters_to_fit_,
        n_parameters_pow2);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

#endif // USE_CUBLAS

void LMFitCUDA::update_states()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    //set up to update the lm_state_gpu_ variable with the Gauss Jordan results
    threads.x = std::min(n_fits_, 256);
    blocks.x = int(std::ceil(REAL(n_fits_) / REAL(threads.x)));

    //update the gpu_data_.states_ variable
    cuda_update_state_after_solving <<< blocks, threads >>>(
        n_fits_,
        gpu_data_.solution_info_,
        gpu_data_.finished_,
        gpu_data_.states_);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

void LMFitCUDA::scale_hessians()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    threads.x = info_.n_parameters_to_fit_*info_.n_fits_per_block_;
    blocks.x = n_fits_ / info_.n_fits_per_block_;

    cuda_modify_step_widths <<< blocks, threads >>>(
        gpu_data_.hessians_,
        gpu_data_.lambdas_,
        gpu_data_.scaling_vectors_,
        info_.n_parameters_to_fit_,
        gpu_data_.iteration_failed_,
        gpu_data_.finished_,
        info_.n_fits_per_block_);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

void LMFitCUDA::project_parameters_to_box()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    threads.x = info_.n_parameters_to_fit_*info_.n_fits_per_block_;
    blocks.x = n_fits_ / info_.n_fits_per_block_;

    cuda_project_parameters_to_box <<< blocks, threads >>>(
        gpu_data_.parameters_,
        info_.n_parameters_,
        info_.n_parameters_to_fit_,
        gpu_data_.parameters_to_fit_indices_,
        gpu_data_.constraints_,
        gpu_data_.constraint_types_,
        gpu_data_.finished_,
        info_.n_fits_per_block_);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

void LMFitCUDA::update_parameters()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    threads.x = info_.n_parameters_*info_.n_fits_per_block_;
    blocks.x = n_fits_ / info_.n_fits_per_block_;

    cuda_update_parameters <<< blocks, threads >>>(
        gpu_data_.parameters_,
        gpu_data_.prev_parameters_,
        gpu_data_.deltas_,
        info_.n_parameters_to_fit_,
        gpu_data_.parameters_to_fit_indices_,
        gpu_data_.finished_,
        info_.n_fits_per_block_);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

void LMFitCUDA::calc_curve_values()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    threads.x = info_.n_points_ * info_.n_fits_per_block_ / info_.n_blocks_per_fit_;

    if (info_.n_blocks_per_fit_ > 1)
        threads.x += info_.n_points_ % threads.x;

    blocks.x = n_fits_ / info_.n_fits_per_block_ * info_.n_blocks_per_fit_;

    cuda_calc_curve_values <<< blocks, threads >>>(
        gpu_data_.parameters_,
        n_fits_,
        info_.n_points_,
        info_.n_parameters_,
        gpu_data_.finished_,
        gpu_data_.values_,
        gpu_data_.derivatives_,
        info_.n_fits_per_block_,
        info_.n_blocks_per_fit_,
        info_.model_id_,
        gpu_data_.chunk_index_,
        gpu_data_.user_info_,
        info_.user_info_size_);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

void LMFitCUDA::calc_chi_squares()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    threads.x = info_.power_of_two_n_points_ * info_.n_fits_per_block_ / info_.n_blocks_per_fit_;
    blocks.x = n_fits_ / info_.n_fits_per_block_ * info_.n_blocks_per_fit_;

    int const shared_size = sizeof(REAL) * threads.x;

    REAL * chi_squares = 
        info_.n_blocks_per_fit_ > 1 ? gpu_data_.subtotals_ : gpu_data_.chi_squares_;

    cuda_calculate_chi_squares <<< blocks, threads, shared_size >>>(
        chi_squares,
        gpu_data_.states_,
        gpu_data_.data_,
        gpu_data_.values_,
        gpu_data_.weights_,
        info_.n_points_,
        n_fits_,
        info_.estimator_id_,
        gpu_data_.finished_,
        info_.n_fits_per_block_,
        gpu_data_.user_info_,
        info_.user_info_size_);
    CUDA_CHECK_STATUS(cudaGetLastError());

    threads.x = std::min(n_fits_, 256);
    blocks.x = int(std::ceil(REAL(n_fits_) / REAL(threads.x)));

    if (info_.n_blocks_per_fit_ > 1)
    {
        cuda_sum_chi_square_subtotals <<< blocks, threads >>> (
            gpu_data_.chi_squares_,
            gpu_data_.subtotals_,
            info_.n_blocks_per_fit_,
            n_fits_,
            gpu_data_.finished_);
        CUDA_CHECK_STATUS(cudaGetLastError());
    }

    cuda_check_fit_improvement <<< blocks, threads >>>(
        gpu_data_.iteration_failed_,
        gpu_data_.chi_squares_,
        gpu_data_.prev_chi_squares_,
        n_fits_,
        gpu_data_.finished_);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

void LMFitCUDA::calc_gradients()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    threads.x = info_.power_of_two_n_points_ * info_.n_fits_per_block_ / info_.n_blocks_per_fit_;
    blocks.x = n_fits_ / info_.n_fits_per_block_ * info_.n_blocks_per_fit_;

    int const shared_size = sizeof(REAL) * threads.x;

    REAL * gradients
        = info_.n_blocks_per_fit_ > 1 ? gpu_data_.subtotals_ : gpu_data_.gradients_;

    cuda_calculate_gradients <<< blocks, threads, shared_size >>>(
        gradients,
        gpu_data_.data_,
        gpu_data_.values_,
        gpu_data_.derivatives_,
        gpu_data_.weights_,
        info_.n_points_,
        n_fits_,
        info_.n_parameters_,
        info_.n_parameters_to_fit_,
        gpu_data_.parameters_to_fit_indices_,
        info_.estimator_id_,
        gpu_data_.finished_,
        gpu_data_.iteration_failed_,
        info_.n_fits_per_block_,
        gpu_data_.user_info_,
        info_.user_info_size_);
    CUDA_CHECK_STATUS(cudaGetLastError());

    if (info_.n_blocks_per_fit_ > 1)
    {
        int const gradients_size = n_fits_ * info_.n_parameters_to_fit_;
        threads.x = std::min(gradients_size, 256);
        blocks.x = int(std::ceil(REAL(gradients_size) / REAL(threads.x)));

        cuda_sum_gradient_subtotals <<< blocks, threads >>> (
            gpu_data_.gradients_,
            gpu_data_.subtotals_,
            info_.n_blocks_per_fit_,
            n_fits_,
            info_.n_parameters_to_fit_,
            gpu_data_.iteration_failed_,
            gpu_data_.finished_);
        CUDA_CHECK_STATUS(cudaGetLastError());
    }
}

void LMFitCUDA::calc_hessians()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    int const n_unique_values
        = info_.n_parameters_to_fit_ * (info_.n_parameters_to_fit_ + 1) / 2;

    int n_hessians_per_block = 1;

    if (info_.n_parameters_to_fit_)
    {
        while ((n_hessians_per_block + 1) * n_unique_values < info_.warp_size_)
        {
            n_hessians_per_block++;
        }
    }

    int const temp_threads_x = n_unique_values * n_hessians_per_block;

    threads.x = std::min(temp_threads_x, info_.max_threads_);
    
    blocks.y
        = temp_threads_x / info_.max_threads_
        + int((temp_threads_x % info_.max_threads_) > 0);
    
    blocks.x
        = n_fits_ / n_hessians_per_block
        + int((n_fits_ % n_hessians_per_block) > 0);

    cuda_calculate_hessians <<< blocks, threads >>>(
        gpu_data_.hessians_,
        gpu_data_.data_,
        gpu_data_.values_,
        gpu_data_.derivatives_,
        gpu_data_.weights_,
        n_fits_,
        info_.n_points_,
        info_.n_parameters_,
        info_.n_parameters_to_fit_,
        gpu_data_.parameters_to_fit_indices_,
        info_.estimator_id_,
        gpu_data_.iteration_failed_,
        gpu_data_.finished_,
        gpu_data_.user_info_,
        info_.user_info_size_);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

void LMFitCUDA::evaluate_iteration(int const iteration)
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    threads.x = std::min(n_fits_, 256);
    blocks.x = int(std::ceil(REAL(n_fits_) / REAL(threads.x)));

    cuda_check_for_convergence<<< blocks, threads >>>(
        gpu_data_.finished_,
        tolerance_,
        gpu_data_.states_,
        gpu_data_.chi_squares_,
        gpu_data_.prev_chi_squares_,
        iteration,
        info_.max_n_iterations_,
        n_fits_);
    CUDA_CHECK_STATUS(cudaGetLastError());

    gpu_data_.set(gpu_data_.all_finished_, 1);

    cuda_evaluate_iteration<<< blocks, threads >>>(
        gpu_data_.all_finished_,
        gpu_data_.n_iterations_,
        gpu_data_.finished_,
        iteration,
        gpu_data_.states_,
        n_fits_);
    CUDA_CHECK_STATUS(cudaGetLastError());

    gpu_data_.read(&all_finished_, gpu_data_.all_finished_);

    cuda_prepare_next_iteration<<< blocks, threads >>>(
        gpu_data_.lambdas_,
        gpu_data_.chi_squares_,
        gpu_data_.prev_chi_squares_,
        gpu_data_.parameters_,
        gpu_data_.prev_parameters_,
        n_fits_,
        info_.n_parameters_);
    CUDA_CHECK_STATUS(cudaGetLastError());
}
