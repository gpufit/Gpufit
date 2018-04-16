#ifndef GPUFIT_MLE_CUH_INCLUDED
#define GPUFIT_MLE_CUH_INCLUDED

#include <math.h>

/* Description of the calculate_chi_square_mle function
* =====================================================
*
* This function calculates the chi-square values for the MLE estimator.
*
* Parameters:
*
* chi_square: An output vector of chi-square values for each data point.
*
* point_index: The data point index.
*
* data: An input vector of data.
*
* value: An input vector of fitting curve values.
*
* weight: An input vector of values for weighting chi-square values. It is not used
*         in this function. It can be used in functions calculating other estimators
*         than the MLE, such as LSE.
*
* state: A pointer to a value which indicates whether the fitting process was carreid
*        out correctly or which problem occurred. It is set to 3 if a fitting curve
*        value is negative.
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The number of elements in user_info. (not used)
*
* Calling the calculate_chi_square_mle function
* =============================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_chi_square_mle(
    volatile REAL * chi_square,
    int const point_index,
    REAL const * data,
    REAL const * value,
    REAL const * weight,
    int * state,
    char * user_info,
    std::size_t const user_info_size)
{
    if (value[point_index] < 0)
    {
        *state = 3;
    }

    REAL const deviation = value[point_index] - data[point_index];

    if (data[point_index] != 0)
    {
        chi_square[point_index]
            = 2 * (deviation - data[point_index] * std::log(value[point_index] / data[point_index]));
    }
    else
    {
        chi_square[point_index] = 2 * deviation;
    }
}

/* Description of the calculate_hessian_mle function
* ==================================================
*
* This function calculates the hessian matrix values of the MLE estimator. The
* calculation is performed based on previously calculated derivative values.
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
* data: An input vector of data values.
*
* value: An input vector of fitting curve values.
*
* derivative: An input vector of partial derivative values of the fitting
*             curve with respect to the fitting parameters for each data point.
*
* weight: An input vector of values for weighting hessian matrix values. It is not
*         used in this function. It can be used in functions calculating other estimators
*         than the MLE, such as LSE.
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The number of elements in user_info. (not used)
*
* Calling the calculate_hessian_mle function
* ==========================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_hessian_mle(
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
    *hessian
        += data[point_index]
        / (value[point_index] * value[point_index])
        * derivative[parameter_index_i] * derivative[parameter_index_j];
}

/* Description of the calculate_gradient_mle function
* ===================================================
*
* This function calculates the gradient values of the MLE estimator based
* on previously calculated derivative values.
*
* Parameters:
*
* gradient: An output vector of values of the gradient vector for each data point.
*
* point_index: The data point index.
*
* parameter_index: The parameter index.
*
* data: An input vector of data values.
*
* value: An input vector of fitting curve values.
*
* derivative: An input vector of partial derivative values of the fitting
*             curve with respect to the fitting parameters for each data point.
*
* weight: An input vector of values for weighting gradient vector values. It is not
*         used in this function. It can be used in functions calculating other estimators
*         than the MLE, such as LSE.
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The number of elements in user_info. (not used)
*
* Calling the calculate_gradient_mle function
* ===========================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_gradient_mle(
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
    gradient[point_index]
        = -derivative[parameter_index]
        * (1 - data[point_index] / value[point_index]);
}

#endif
