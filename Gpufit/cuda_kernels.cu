#include "constants.h"
#include "cuda_kernels.cuh"
#include "models/models.cuh"
#include "estimators/estimators.cuh"

/* Description of the cuda_calc_curve_values function
* ===================================================
*
* This function calls one of the fitting curve functions depending on the input
* parameter model_id. The fitting curve function calculates the values of
* the fitting curves and its partial derivatives with respect to the fitting
* curve parameters. Multiple fits are calculated in parallel.
*
* Parameters:
*
* parameters: An input vector of concatenated sets of model parameters.
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* n_parameters: The number of curve parameters.
*
* finished: An input vector which allows the calculation to be skipped for single
*           fits.
*
* values: An output vector of concatenated sets of model function values.
*
* derivatives: An output vector of concatenated sets of model function partial
*              derivatives.
*
* n_fits_per_block: The number of fits calculated by each thread block.
*
* n_blocks_per_fit: The number of thread blocks used to calculate one fit.
*
* model_id: The fitting model ID.
*
* chunk_index: The data chunk index.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the cuda_calc_curve_values function
* ===========================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   threads.x = n_points * n_fits_per_block / n_blocks_per_fit;
*   blocks.x = n_fits / n_fits_per_block * n_blocks_per_fit;
*
*   cuda_calc_curve_values<<< blocks, threads >>>(
*       parameters,
*       n_fits,
*       n_points,
*       n_parameters,
*       finished,
*       values,
*       derivatives,
*       n_fits_per_block,
*       n_blocks_per_fit,
*       model_id,
*       chunk_index,
*       user_info,
*       user_info_size);
*
*/

__global__ void cuda_calc_curve_values(
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
    std::size_t const user_info_size)
{
    int const fit_in_block = threadIdx.x / n_points;
    int const fit_index = blockIdx.x * n_fits_per_block / n_blocks_per_fit + fit_in_block;
    int const fit_piece = blockIdx.x % n_blocks_per_fit;
    int const point_index = threadIdx.x - fit_in_block * n_points + fit_piece * blockDim.x;
    int const first_point = fit_index * n_points;

    REAL * current_values = values + first_point;
    REAL * current_derivatives = derivatives + first_point * n_parameters;
    REAL const * current_parameters = parameters + fit_index * n_parameters;

    if (finished[fit_index])
        return;
    if (point_index >= n_points)
        return;

    calculate_model(model_id, current_parameters, n_fits, n_points, current_values, current_derivatives, point_index, fit_index, chunk_index, user_info, user_info_size);
}

/* Description of the sum_up_floats function
* ==========================================
*
* This function sums up a vector of REAL values and stores the result at the
* first place of the vector.
*
* Parameters:
*
* shared_array: An input vector of REAL values. The vector must be stored
*               on the shared memory of the GPU. The size of this vector must be a
*               power of two. Use zero padding to extend it to the next highest
*               power of 2 greater than the number of elements.
*
* size: The number of elements in the input vector considering zero padding.
*
* Calling the sum_up_floats function
* ==================================
*
* This __device__ function can be only called from a __global__ function or
* an other __device__ function. When calling the function, the blocks and threads
* of the __global__ function must be set up correctly, as shown in the following
* example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   threads.x = size * vectors_per_block;
*   blocks.x = n_vectors / vectors_per_block;
*
*   global_function<<< blocks, threads >>>(parameter1, ...);
*
*/

__device__ void sum_up_floats(volatile REAL* shared_array, int const size)
{
    int const fit_in_block = threadIdx.x / size;
    int const point_index = threadIdx.x - (fit_in_block*size);

    int current_n_points = size >> 1;
    __syncthreads();
    while (current_n_points)
    {
        if (point_index < current_n_points)
        {
            shared_array[point_index] += shared_array[point_index + current_n_points];
        }
        current_n_points >>= 1;
        __syncthreads();
    }
}

/* Description of the cuda_sum_chi_square_subtotals function
* ==========================================================
*
* This function sums up chi_square subtotals in place.
*
* Parameters:
*
* chi_squares: A vector of chi-square values for multiple fits.
*              in: subtotals
*              out: totals
*
* n_blocks_per_fit: The number of blocks used to calculate one fit. It is
*                   equivalent to the number of subtotals per fit.
*
* n_fits: The number of fits.
*
* finished: An input vector which allows the calculation to be skipped
*           for single fits.
*
* Calling the cuda_sum_chi_square_subtotals function
* ==================================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   int const example_value = 256;
*
*   threads.x = min(n_fits, example_value);
*   blocks.x = int(ceil(REAL(n_fits) / REAL(threads.x)));
*
*   cuda_sum_chi_square_subtotals<<< blocks, threads >>>(
*       chi_squares,
*       n_blocks_per_fit,
*       n_fits,
*       finished);
*
*/

__global__ void cuda_sum_chi_square_subtotals(
    REAL * chi_squares,
    REAL const * subtotals,
    int const n_blocks_per_fit,
    int const n_fits,
    int const * finished)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n_fits || finished[index])
        return;

    REAL * chi_square = chi_squares + index;
    REAL const * subtotal = subtotals + index;

    double sum = 0.0;
    for (int i = 0; i < n_blocks_per_fit; i++)
        sum += subtotal[i * n_fits];

    chi_square[0] = sum;
}

/* Description of the cuda_check_fit_improvement function
* =======================================================
*
* This function checks after each calculation of chi-square values whether the
* currently calculated chi-square values are lower than chi-square values calculated
* in the previous iteration and sets the iteration_failed flags.
*
* Parameters:
*
* iteration_failed: An output vector of flags which indicate whether the fitting
*                   process improved the fit in the last iteration. If yes it is set
*                   to 0 otherwise to 1.
*
* chi_squares: An input vector of chi-square values for multiple fits.
*
* prev_chi_squares: An input vector of chi-square values for multiple fits calculated
*                   in the previous iteration.
*
* n_fits: The number of fits.
*
* finished: An input vector which allows the calculation to be skipped
*           for single fits.
*
* Calling the cuda_check_fit_improvement function
* ===============================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   int const example_value = 256;
*
*   threads.x = min(n_fits, example_value);
*   blocks.x = int(ceil(REAL(n_fits) / REAL(threads.x)));
*
*   cuda_check_fit_improvement <<< blocks, threads >>>(
*       iteration_failed,
*       chi_squares,
*       prev_chi_squares,
*       n_fits,
*       finished);
*
*/

__global__ void cuda_check_fit_improvement(
    int * iteration_failed,
    REAL const * chi_squares,
    REAL const * prev_chi_squares,
    int const n_fits,
    int const * finished)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n_fits || finished[index])
        return;

    bool const prev_chi_squares_initialized = prev_chi_squares[index] != 0.;
    // chi_squares[index] can be NaN which compares to false with any other number
    bool const chi_square_decreased = (chi_squares[index] < prev_chi_squares[index]);
    if (prev_chi_squares_initialized && !chi_square_decreased)
    {
        iteration_failed[index] = 1;
    }
    else
    {
        iteration_failed[index] = 0;
    }
}

/* Description of the cuda_calculate_chi_squares function
* ========================================================
*
* This function calls one of the estimator funktions depending on the input
* parameter estimator_id. The estimator function calculates the chi-square values.
* The calcluation is performed for multiple fits in parallel.
*
* Parameters:
*
* chi_squares: An output vector of concatenated chi-square values.
*
* states: An output vector of values which indicate whether the fitting process
*         was carreid out correctly or which problem occurred. In this function
*         it is only used for MLE. It is set to 3 if a fitting curve value is
*         negative. This vector includes the states for multiple fits.
*
* data: An input vector of data for multiple fits
*
* values: An input vector of concatenated sets of model function values.
*
* weights: An input vector of values for weighting chi-square, gradient and hessian,
*          while using LSE
*
* n_points: The number of data points per fit.
*
* n_fits: The number of fits.
*
* estimator_id: The estimator ID.
*
* finished: An input vector which allows the calculation to be skipped for single
*           fits.
*
* n_fits_per_block: The number of fits calculated by each thread block.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the cuda_calculate_chi_squares function
* ================================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   threads.x = power_of_two_n_points * n_fits_per_block / n_blocks_per_fit;
*   blocks.x = n_fits / n_fits_per_block * n_blocks_per_fit;
*
*   int const shared_size = sizeof(REAL) * threads.x;
*
*   cuda_calculate_chi_squares<<< blocks, threads, shared_size >>>(
*       chi_squares,
*       states,
*       data,
*       values,
*       weights,
*       n_points,
*       n_fits,
*       estimator_id,
*       finished,
*       n_fits_per_block,
*       user_info,
*       user_info_size);
*
*/

__global__ void cuda_calculate_chi_squares(
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
    std::size_t const user_info_size)
{
    int const shared_size = blockDim.x / n_fits_per_block;
    int const fit_in_block = threadIdx.x / shared_size;
    int const fit_piece = blockIdx.x / n_fits;
    int const fit_index = blockIdx.x * n_fits_per_block + fit_in_block - fit_piece * n_fits;
    int const point_index = threadIdx.x - fit_in_block * shared_size + fit_piece * shared_size;
    int const first_point = fit_index * n_points;

    if (finished[fit_index])
    {
        return;
    }

    REAL const * current_data = &data[first_point];
    REAL const * current_weight = weights ? &weights[first_point] : NULL;
    REAL const * current_value = &values[first_point];
    int * current_state = &states[fit_index];

    extern __shared__ REAL extern_array[];

    volatile REAL * shared_chi_square
        = extern_array + (fit_in_block - fit_piece) * shared_size;

    if (point_index >= n_points)
    {
        shared_chi_square[point_index] = 0.;
    }

    if (point_index < n_points)
    {
        calculate_chi_square(
            estimator_id,
            shared_chi_square,
            point_index,
            current_data,
            current_value,
            current_weight,
            current_state,
            user_info,
            user_info_size);
    }
    shared_chi_square += fit_piece * shared_size;
    sum_up_floats(shared_chi_square, shared_size);
    chi_squares[fit_index + fit_piece * n_fits] = shared_chi_square[0];
}

/* Description of the cuda_sum_gradient_subtotals function
* ========================================================
*
* This function sums up the chi-square gradient subtotals in place.
*
* Parameters:
*
* gradients: A vector of gradient values for multiple fits.
*            in: subtotals
*            out: totals
*
* n_blocks_per_fit: The number of blocks used to calculate one fit
*
* n_fits: The number of fits.
*
* n_parameters_to_fit: The number of model parameters, that are not held fixed.
*
* skip: An input vector which allows the calculation to be skipped for single fits.
*
* finished: An input vector which allows the calculation to be skipped for single
*           fits.
*
* Calling the cuda_sum_gradient_subtotals function
* ================================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   int const example_value = 256;
*
*   threads.x = min(n_fits, example_value);
*   blocks.x = int(ceil(REAL(n_fits) / REAL(threads.x)));
*
*   cuda_sum_gradient_subtotals<<< blocks,threads >>>(
*       gradients,
*       n_blocks_per_fit,
*       n_fits,
*       n_parameters_to_fit,
*       skip,
*       finished);
*
*/

__global__ void cuda_sum_gradient_subtotals(
    REAL * gradients,
    REAL const * subtotals,
    int const n_blocks_per_fit,
    int const n_fits,
    int const n_parameters,
    int const * skip,
    int const * finished)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    int const fit_index = index / n_parameters;

    if (fit_index >= n_fits || finished[fit_index] || skip[fit_index])
        return;

    REAL * gradient = gradients + index;
    REAL const * subtotal = subtotals + index;

    double sum = 0.0;
    for (int i = 0; i < n_blocks_per_fit; i++)
        sum += subtotal[i * n_fits * n_parameters];

    gradient[0] = sum;
}

/* Description of the cuda_calculate_gradients function
* =====================================================
*
* This function calls one of the gradient functions depending on the input
* parameter estimator_id. The gradient function calculates the gradient values
* of the chi-square function calling a __device__ function. The calcluation is
* performed for multiple fits in parallel.
*
* Parameters:
*
* gradients: An output vector of concatenated sets of gradient vector values.
*
* data: An input vector of data for multiple fits
*
* values: An input vector of concatenated sets of model function values.
*
* derivatives: An input vector of concatenated sets of model function partial
*              derivatives.
*
* weights: An input vector of values for weighting chi-square, gradient and hessian,
*          while using LSE
*
* n_points: The number of data points per fit.
*
* n_fits: The number of fits.
*
* n_parameters: The number of fitting curve parameters.
*
* n_parameters_to_fit: The number of fitting curve parameters, that are not held
*                      fixed.
*
* parameters_to_fit_indices: An input vector of indices of fitting curve parameters,
*                            that are not held fixed.
*
* estimator_id: The estimator ID.
*
* finished: An input vector which allows the calculation to be skipped for single
*           fits.
*
* skip: An input vector which allows the calculation to be skipped for single fits.
*
* n_fits_per_block: The number of fits calculated by each thread block.
*
* user_info: An input vector containing user information.
*
* user_info_size: The number of elements in user_info.
*
* Calling the cuda_calculate_gradients function
* =============================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   threads.x = power_of_two_n_points * n_fits_per_block / n_blocks_per_fit;
*   blocks.x = n_fits / n_fits_per_block * n_blocks_per_fit;
*
*   int const shared_size = sizeof(REAL) * threads.x;
*
*   cuda_calculate_gradients<<< blocks, threads, shared_size >>>(
*       gradients,
*       data,
*       values,
*       derivatives,
*       weight,
*       n_points,
*       n_fits,
*       n_parameters,
*       n_parameters_to_fit,
*       parameters_to_fit_indices,
*       estimator_id,
*       finished,
*       skip,
*       n_fits_per_block,
*       user_info,
*       user_info_size);
*
*/

__global__ void cuda_calculate_gradients(
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
    std::size_t const user_info_size)
{
    int const shared_size = blockDim.x / n_fits_per_block;
    int const fit_in_block = threadIdx.x / shared_size;
    int const fit_piece = blockIdx.x / n_fits;
    int const fit_index = blockIdx.x * n_fits_per_block + fit_in_block - fit_piece * n_fits;
    int const point_index = threadIdx.x - fit_in_block * shared_size + fit_piece * shared_size;
    int const first_point = fit_index * n_points;

    if (finished[fit_index] || skip[fit_index])
    {
        return;
    }

    REAL const * current_data = &data[first_point];
    REAL const * current_weight = weights ? &weights[first_point] : NULL;
    REAL const * current_derivative = &derivatives[first_point * n_parameters];
    REAL const * current_value = &values[first_point];

    extern __shared__ REAL extern_array[];

    volatile REAL * shared_gradient = extern_array + (fit_in_block - fit_piece) * shared_size;

    if (point_index >= n_points)
    {
        shared_gradient[point_index] = 0.;
    }

    for (int parameter_index = 0; parameter_index < n_parameters_to_fit; parameter_index++)
    {
        if (point_index < n_points)
        {
            int const derivative_index = parameters_to_fit_indices[parameter_index] * n_points + point_index;

            calculate_gradient(
                estimator_id,
                shared_gradient,
                point_index,
                derivative_index,
                current_data,
                current_value,
                current_derivative,
                current_weight,
                user_info,
                user_info_size);
        }
        sum_up_floats(shared_gradient + fit_piece * shared_size, shared_size);
        gradients[(fit_index * n_parameters_to_fit + parameter_index) + fit_piece * n_fits * n_parameters_to_fit]
            = shared_gradient[fit_piece * shared_size];
    }
}

/* Description of the cuda_calculate_hessians function
* ====================================================
*
* This function calls one of the hessian function depending on the input
* parameter estimator_id. The hessian funcion calculates the hessian matrix
* values of the chi-square function calling a __device__ functions. The
* calcluation is performed for multiple fits in parallel.
*
* Parameters:
*
* hessians: An output vector of concatenated sets of hessian matrix values.
*
* data: An input vector of data for multiple fits
*
* values: An input vector of concatenated sets of model function values.
*
* derivatives: An input vector of concatenated sets of model function partial
*              derivatives.
*
* weights: An input vector of values for weighting chi-square, gradient and hessian,
*          while using LSE
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* n_parameters: The number of fitting curve parameters.
*
* n_parameters_to_fit: The number of fitting curve parameters, that are not held
*                      fixed.
*
* parameters_to_fit_indices: An input vector of indices of fitting curve parameters,
*                            that are not held fixed.
*
* estimator_id: The estimator ID.
*
* skip: An input vector which allows the calculation to be skipped for single fits.
*
* finished: An input vector which allows the calculation to be skipped for single
*           fits.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the cuda_calculate_hessians function
* ============================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   int n_unique_values = n_parameters_to_fit * (n_parameters_to_fit + 1) / 2;
*
*   threads.x
*       = min(n_unique_values * n_fits_per_block, max_threads_per_block);
*
*   blocks.y
*       = threads.x / max_threads_per_block
*       + int((threads.x % max_threads_per_block) > 0);
*
*   blocks.x
*       = n_fits / n_fits_per_block
*       + int((n_fits % n_fits_per_block) > 0);
*
*   cuda_calculate_hessians<<< blocks, threads >>>(
*       hessians,
*       data,
*       values,
*       derivatives,
*       weight,
*       n_fits,
*       n_points,
*       n_parameters,
*       n_parameters_to_fit,
*       parameters_to_fit_indices,
*       estimator_id,
*       skip,
*       finished,
*       user_info,
*       user_info_size);
*
*/

__global__ void cuda_calculate_hessians(
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
    std::size_t const user_info_size)
{
    int const n_unique_values = n_parameters_to_fit * (n_parameters_to_fit + 1) / 2;
    int const n_fits_per_block = blockDim.x * gridDim.y / n_unique_values;
    
    int const fit_in_block
        = (gridDim.y == 1)
        ? (blockIdx.y * blockDim.x + threadIdx.x) / n_unique_values
        : 0;

    int const fit_index = blockIdx.x * n_fits_per_block + fit_in_block;

    if (fit_index >= n_fits || finished[fit_index] || skip[fit_index])
    {
        return;
    }

    int const first_point = fit_index * n_points;
    int const parameter_index = (blockIdx.y * blockDim.x + threadIdx.x) - fit_in_block * n_unique_values;

    if (parameter_index >= n_unique_values)
    {
        return;
    }

    int const parameter_index_i
        = n_parameters_to_fit
        - 1.
        - std::floor(
            .5*(
                std::sqrt(
                    - 8. * (parameter_index - n_parameters_to_fit)
                    + 4. * n_parameters_to_fit * (n_parameters_to_fit - 1.)
                    - 7.
                ) - 1.
            )
        );

    int const parameter_index_j
        = parameter_index
        + parameter_index_i
        - parameter_index_i*(n_parameters_to_fit - (parameter_index_i - 1) / 2.);

    REAL * current_hessian = &hessians[fit_index * n_parameters_to_fit * n_parameters_to_fit];
    REAL const * current_data = &data[first_point];
    REAL const * current_weight = weights ? &weights[first_point] : NULL;
    REAL const * current_derivative = &derivatives[first_point*n_parameters];
    REAL const * current_value = &values[first_point];

    int const hessian_index_ij = parameter_index_i * n_parameters_to_fit + parameter_index_j;
    int const hessian_index_ji = parameter_index_j * n_parameters_to_fit + parameter_index_i;
    int const derivative_index_i = parameters_to_fit_indices[parameter_index_i] * n_points;
    int const derivative_index_j = parameters_to_fit_indices[parameter_index_j] * n_points;

    double sum = 0.0;
    for (int point_index = 0; point_index < n_points; point_index++)
    {
        calculate_hessian(
            estimator_id,
            &sum,
            point_index,
            derivative_index_i + point_index,
            derivative_index_j + point_index,
            current_data,
            current_value,
            current_derivative,
            current_weight,
            user_info,
            user_info_size);
    }
    current_hessian[hessian_index_ij] = sum;
    current_hessian[hessian_index_ji] = sum;
}

/* Description of the cuda_modify_step_widths function
* ====================================================
*
* This function midifies the diagonal elements of the hessian matrices by multiplying
* them by the factor (1+ lambda). This operation controls the step widths of the
* iteration. If the last iteration failed, befor modifying the hessian, the diagonal
* elements of the hessian are calculated back to represent unmodified values.
*
* hessians: An input and output vector of hessian matrices, which are modified by
*           the lambda values.
*
* lambdas: An input vector of values for modifying the hessians.
*
* n_parameters: The number of fitting curve parameters.
*
* iteration_failed: An input vector which indicates whether the previous iteration
*                   failed.
*
* finished: An input vector which allows the calculation to be skipped for single fits.
*
* n_fits_per_block: The number of fits calculated by each thread block.
*
* Calling the cuda_modify_step_widths function
* ============================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   threads.x = n_parameters_to_fit * n_fits_per_block;
*   blocks.x = n_fits / n_fits_per_block;
*
*   cuda_modify_step_width<<< blocks, threads >>>(
*       hessians,
*       lambdas,
*       n_parameters,
*       iteration_failed,
*       finished,
*       n_fits_per_block);
*
*/

__global__ void cuda_modify_step_widths(
    REAL * hessians,
    REAL const * lambdas,
    REAL * scaling_vectors,
    unsigned int const n_parameters,
    int const * iteration_failed,
    int const * finished,
    int const n_fits_per_block)
{
    int const shared_size = blockDim.x / n_fits_per_block;
    int const fit_in_block = threadIdx.x / shared_size;
    int const parameter_index = threadIdx.x - fit_in_block * shared_size;
    int const fit_index = blockIdx.x * n_fits_per_block + fit_in_block;

    if (finished[fit_index])
    {
        return;
    }

    REAL * hessian = &hessians[fit_index * n_parameters * n_parameters];
    REAL * scaling_vector = &scaling_vectors[fit_index * n_parameters];
    REAL const & lambda = lambdas[fit_index];

    int const diagonal_index = parameter_index * n_parameters + parameter_index;

    if (iteration_failed[fit_index])
    {
        hessian[diagonal_index] -= scaling_vector[parameter_index] * lambda / 10.;
    }

    // adaptive scaling
    scaling_vector[parameter_index]
        = max(scaling_vector[parameter_index], hessian[diagonal_index]);

    // continuous scaling
    //scaling_vector[parameter_index] = hessian[diagonal_index];
    
    // initial scaling
    //if (scaling_vector[parameter_index] == 0.)
    //    scaling_vector[parameter_index] = hessian[diagonal_index];

    hessian[diagonal_index] += scaling_vector[parameter_index] * lambda;
}

/* Description of the cuda_update_parameters function
* ===================================================
*
* This function stores the fitting curve parameter values in prev_parameters and
* updates them after each iteration.
*
* Parameters:
*
* parameters: An input and output vector of concatenated sets of model
*             parameters.
*
* prev_parameters: An input and output vector of concatenated sets of model
*                  parameters calculated by the previous iteration.
*
* deltas: An input vector of concatenated delta values, which are added to the
*         model parameters.
*
* n_parameters_to_fit: The number of fitted curve parameters.
*
* parameters_to_fit_indices: The indices of fitted curve parameters.
*
* finished: An input vector which allows the parameter update to be skipped for single fits.
*
* n_fits_per_block: The number of fits calculated by each threadblock.
*
* Calling the cuda_update_parameters function
* ===========================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   threads.x = n_parameters * n_fits_per_block;
*   blocks.x = n_fits / n_fits_per_block;
*
*   cuda_update_parameters<<< blocks, threads >>>(
*       parameters,
*       prev_parameters,
*       deltas,
*       n_parameters_to_fit,
*       parameters_to_fit_indices,
*       finished,
*       n_fits_per_block);
*
*/

__global__ void cuda_update_parameters(
    REAL * parameters,
    REAL * prev_parameters,
    REAL const * parameter_constraints,
    REAL const * deltas,
    int const n_parameters_to_fit,
    int const * parameters_to_fit_indices,
    int const * finished,
    int const n_fits_per_block)
{
    int const n_parameters = blockDim.x / n_fits_per_block;
    int const fit_in_block = threadIdx.x / n_parameters;
    int const parameter_index = threadIdx.x - fit_in_block * n_parameters;
    int const fit_index = blockIdx.x * n_fits_per_block + fit_in_block;

    REAL * current_parameters = &parameters[fit_index * n_parameters];
    REAL * current_prev_parameters = &prev_parameters[fit_index * n_parameters];

    current_prev_parameters[parameter_index] = current_parameters[parameter_index];

    REAL const * current_constraint = parameter_constraints ? &parameter_constraints[fit_index * n_parameters*2] : NULL;

    if (finished[fit_index])
    {
        return;
    }

    if (parameter_index >= n_parameters_to_fit)
    {
        return;
    }

    REAL const * current_deltas = &deltas[fit_index * n_parameters_to_fit];

    current_parameters[parameters_to_fit_indices[parameter_index]] += current_deltas[parameter_index];
    if (current_constraint)
    {
		if (current_parameters[parameters_to_fit_indices[parameter_index]]<current_constraint[parameters_to_fit_indices[parameter_index]*2+0])
		{
			current_parameters[parameters_to_fit_indices[parameter_index]]=current_constraint[parameters_to_fit_indices[parameter_index]*2+0];
		}
		if (current_parameters[parameters_to_fit_indices[parameter_index]]>current_constraint[parameters_to_fit_indices[parameter_index]*2+1])
		{
			current_parameters[parameters_to_fit_indices[parameter_index]]=current_constraint[parameters_to_fit_indices[parameter_index]*2+1];
		}
    }
}

/* Description of the cuda_update_state_after_solving function
 * ===========================================================
 *
 * This function interprets the singular flag vector of the equation system
 * solving function according to this LM implementation.
 *
 * Parameters:
 *
 * n_fits: The number of fits.
 *
 * solution_info: An input vector used to report whether a fit is singular.
 *
 * finished: An input vector which allows the calculation to by skipped for
 *           single fits.
 *
 * gpufit_states: An output vector of values which indicate whether the fitting
 *                process was carreid out correctly or which problem occurred.
 *                If a hessian matrix of a fit is singular, it is set to 2.
 *
 * Calling the cuda_update_state_after_solving function
 * ====================================================
 *
 * When calling the function, the blocks and threads must be set up correctly,
 * as shown in the following example code.
 *
 *   dim3  threads(1, 1, 1);
 *   dim3  blocks(1, 1, 1);
 *
 *   int const example_value = 256;
 *
 *   threads.x = min(n_fits, example_value);
 *   blocks.x = int(ceil(REAL(n_fits) / REAL(threads.x)));
 *
 *   cuda_update_state_after_solving<<< blocks, threads >>>(
 *       n_fits,
 *       solution_info,
 *       finished,
 *       gpufit_states);
 *
 */
    
__global__ void cuda_update_state_after_solving(
    int const n_fits,
    int const * cublas_info,
    int const * finished,
    int * states)
{
    int const fit_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (fit_index >= n_fits)
        return;

    if (finished[fit_index])
        return;

    if (cublas_info[fit_index] != 0)
        states[fit_index] = SINGULAR_HESSIAN;
}
    
/* Description of the cuda_check_for_convergence function
* =======================================================
*
* This function checks after each iteration whether the fits are converged or not.
* It also checks whether the set maximum number of iterations is reached.
*
* Parameters:
*
* finished: An input and output vector which allows the calculation to be skipped
*           for single fits.
*
* tolerance: The tolerance value for the convergence set by user.
*
* states: An output vector of values which indicate whether the fitting process
*         was carreid out correctly or which problem occurred. If the maximum
*         number of iterations is reached without converging, it is set to 1. If
*         the fit converged it keeps its initial value of 0.
*
* chi_squares: An input vector of chi-square values for multiple fits. Used for the
*              convergence check.
*
* prev_chi_squares: An input vector of chi-square values for multiple fits calculated
*                   in the previous iteration. Used for the convergence check.
*
* iteration: The value of the current iteration. It is compared to the value
*            of the maximum number of iteration set by user.
*
* max_n_iterations: The maximum number of iterations set by user.
*
* n_fits: The number of fits.
*
* Calling the cuda_check_for_convergence function
* ===============================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   int const example_value = 256;
*
*   threads.x = min(n_fits, example_value);
*   blocks.x = int(ceil(REAL(n_fits) / REAL(threads.x)));
*
*   cuda_check_for_convergence<<< blocks, threads >>>(
*       finished,
*       tolerance,
*       states,
*       chi_squares,
*       prev_chi_squares,
*       iteration,
*       max_n_iterations,
*       n_fits);
*
*/

__global__ void cuda_check_for_convergence(
    int * finished,
    REAL const tolerance,
    int * states,
    REAL const * chi_squares,
    REAL const * prev_chi_squares,
    int const iteration,
    int const max_n_iterations,
    int const n_fits)
{
    int const fit_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (fit_index >= n_fits)
    {
        return;
    }

    if (finished[fit_index])
    {
        return;
    }

    int const fit_found
        = abs(chi_squares[fit_index] - prev_chi_squares[fit_index])
        < tolerance * max(1., chi_squares[fit_index]);

    int const max_n_iterations_reached = iteration == max_n_iterations - 1;

    if (fit_found)
    {
        finished[fit_index] = 1;
    }
    else if (max_n_iterations_reached)
    {
        states[fit_index] = MAX_ITERATION;
    }
}

/* Description of the cuda_evaluate_iteration function
* ====================================================
*
* This function evaluates the current iteration.
*   - It marks a fit as finished if a problem occured.
*   - It saves the needed number of iterations if a fit finished.
*   - It checks if all fits finished
*
* Parameters:
*
* all_finished: An output flag, that indicates whether all fits finished.
*
* n_iterations: An output vector of needed iterations for each fit.
*
* finished: An input and output vector which allows the evaluation to be skipped
*           for single fits
*
* iteration: The values of the current iteration.
*
* states: An input vector of values which indicate whether the fitting process
*         was carreid out correctly or which problem occurred.
*
* n_fits: The number of fits.
*
* Calling the cuda_evaluate_iteration function
* ============================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   int const example_value = 256;
*
*   threads.x = min(n_fits, example_value);
*   blocks.x = int(ceil(REAL(n_fits) / REAL(threads.x)));
*
*   cuda_evaluate_iteration<<< blocks, threads >>>(
*       all_finished,
*       n_iterations,
*       finished,
*       iteration,
*       states,
*       n_fits);
*
*/

__global__ void cuda_evaluate_iteration(
    int * all_finished,
    int * n_iterations,
    int * finished,
    int const iteration,
    int const * states,
    int const n_fits)
{
    int const fit_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (fit_index >= n_fits)
    {
        return;
    }

    if (states[fit_index] != CONVERGED)
    {
        finished[fit_index] = 1;
    }

    if (finished[fit_index] && n_iterations[fit_index] == 0)
    {
        n_iterations[fit_index] = iteration + 1;
    }

    if (!finished[fit_index])
    {
        *all_finished = 0;
    }
}

/* Description of the cuda_prepare_next_iteration function
* ========================================================
*
* This function prepares the next iteration. It either updates previous
* chi-square values or sets currently calculated chi-square values and
* parameters to values calculated by the previous iteration. This function also
* updates lambda values.
*
* Parameters:
*
* lambdas: An output vector of values which control the step width by modifying
*          the diagonal elements of the hessian matrices.
*
* chi_squares: An input and output vector of chi-square values for multiple fits.
*
* prev_chi_squares: An input and output vector of chi-square values for multiple
*                   fits calculated in the previous iteration.
*
* parameters: An output vector of concatenated sets of model parameters.
*
* prev_parameters: An input vector of concatenated sets of model parameters
*                  calculated in the previous iteration.
*
* n_fits: The number of fits.
*
* n_parameters: The number of fitting curve parameters.
*
* Calling the cuda_prepare_next_iteration function
* ================================================
*
* When calling the function, the blocks and threads must be set up correctly,
* as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   int const example_value = 256;
*
*   threads.x = min(n_fits, example_value);
*   blocks.x = int(ceil(REAL(n_fits) / REAL(threads.x)));
*
*   cuda_prepare_next_iteration<<< blocks, threads >>>(
*       lambdas,
*       chi_squares,
*       prev_chi_squares,
*       parameters,
*       prev_parameters,
*       n_fits,
*       n_parameters);
*
*/

__global__ void cuda_prepare_next_iteration(
    REAL * lambdas,
    REAL * chi_squares,
    REAL * prev_chi_squares,
    REAL * parameters,
    REAL const * prev_parameters,
    int const n_fits,
    int const n_parameters)
{
    int const fit_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (fit_index >= n_fits)
    {
        return;
    }

    if (chi_squares[fit_index] < prev_chi_squares[fit_index])
    {
        lambdas[fit_index] *= 0.1f;
        prev_chi_squares[fit_index] = chi_squares[fit_index];
    }
    else
    {
        lambdas[fit_index] *= 10.;
        chi_squares[fit_index] = prev_chi_squares[fit_index];
        for (int iparameter = 0; iparameter < n_parameters; iparameter++)
        {
            parameters[fit_index * n_parameters + iparameter] = prev_parameters[fit_index * n_parameters + iparameter];
        }
    }
}
