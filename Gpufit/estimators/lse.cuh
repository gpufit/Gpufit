#ifndef GPUFIT_LSE_CUH_INCLUDED
#define GPUFIT_LSE_CUH_INCLUDED

/* Description of the calculate_chi_square_lse function
* =====================================================
*
* This function calculates the chi-square values for the weighted LSE estimator.
*
* Parameters:
*
* chi_square: An output vector of chi-square values for each data point.
*
* point_index: The data point index.
*
* data: An input vector of data values.
*
* value: An input vector of fitting curve values.
*
* weight: An optional input vector of values for weighting the chi-square values.
*
* state: A pointer to a value which indicates whether the fitting
*        process was carreid out correctly or which problem occurred.
*        In this function it is not used. It can be used in functions calculating
*        other estimators than the LSE, such as MLE. It is passed into this function
*        to provide the same interface for all estimator functions.
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The number of elements in user_info. (not used)
*
* Calling the calculate_chi_square_lse function
* =============================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_chi_square_lse(
    volatile REAL * chi_square,
    int const point_index,
    REAL const * data,
    REAL const * value,
    REAL const * weight,
    int * state,
    char * user_info,
    std::size_t const user_info_size)
{
    REAL const deviation = value[point_index] - data[point_index];

    if (weight)
    {
        chi_square[point_index] = deviation * deviation * weight[point_index];
    }
    else
    {
        chi_square[point_index] = deviation * deviation;
    }
}

/* Description of the calculate_hessian_lse function
* ==================================================
*
* This function calculates the hessian matrix values of the weighted LSE estimator.
* The calculation is performed based on previously calculated fitting curve derivative
* values.
*
* Parameters:
*
* hessian: An output vector of values of the hessian matrix for each data point.
*
* point_index: The data point index.
*
* parameter_index_i: Index of the hessian column.
*
* parameter_index_j: Index of the hessian row.
*
* data: An input vector of data values. (not used)
*
* value: An input vector of fitting curve values. (not used)
*
* derivative: An input vector of partial derivative values of the fitting
*             curve with respect to the fitting parameters for each data point.
*
* weight: An optional input vector of values for weighting the hessian matrix values.
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The number of elements in user_info. (not used)
*
* Calling the calculate_hessian_lse function
* ==========================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_hessian_lse(
    double * hessian,
    int const point_index,
    int const parameter_index_i,
    int const parameter_index_j,
    REAL const * data,
    REAL const * value,
    REAL const * derivative,
    REAL const * weight,
    char * user_info,
    std::size_t const user_info_size)
{
    if (weight)
    {
        *hessian
            += derivative[parameter_index_i] * derivative[parameter_index_j]
            * weight[point_index];
    }
    else
    {
        *hessian
            += derivative[parameter_index_i] * derivative[parameter_index_j];
    }
}

/* Description of the calculate_gradient_lse function
* ===================================================
*
* This function calculates the gradient values of the weighted LSE estimator
* based on previously calculated fitting curve derivative values.
*
* Parameters:
*
* gradient: An output vector of values of the gradient vector for each data point.
*
* point_index: The data point index.
*
* parameter_index: The parameter index.
*
* n_parameters: The number of fitting curve parameters.
*
* data: An input vector of data values.
*
* value: An input vector of fitting curve values.
*
* derivative: An input vector of partial derivative values of the fitting
*             curve with respect to the fitting parameters for each data point.
*
* weight: An optional input vector of values for weighting gradient values.
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The number of elements in user_info. (not used)
*
* Calling the calculate_gradient_lse function
* ===========================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_gradient_lse(
    volatile REAL * gradient,
    int const point_index,
    int const parameter_index,
    REAL const * data,
    REAL const * value,
    REAL const * derivative,
    REAL const * weight,
    char * user_info,
    std::size_t const user_info_size)
{
    REAL const deviation = data[point_index] - value[point_index];

    if (weight)
    {
        gradient[point_index]
            = derivative[parameter_index] * deviation * weight[point_index];
    }
    else
    {
        gradient[point_index]
            = derivative[parameter_index] * deviation;
    }
}

#endif
