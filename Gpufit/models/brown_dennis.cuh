#ifndef GPUFIT_BROWNDENNIS_CUH_INCLUDED
#define GPUFIT_BROWNDENNIS_CUH_INCLUDED

/* Description of the calculate_brown_dennis function
* ===================================================
*
* This function calculates the values of a one-dimensional function defined
* in [Brown, K. M. & Dennis J. E. New computational algorithms for
* minimizing a sum of squares of nonlinear functions. Department of Computer
* Science report 71-6 (1971)] and their partial derivatives with respect to
* the model parameters. 
*
* Parameters:
*
* parameters: An input vector of model parameters.
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The size of user_info in bytes. (not used)
*
* Calling the calculate_brown_dennis function
* ===========================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_brown_dennis(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // indices

    REAL const x = (REAL)point_index / 5.f;

    // parameters

    REAL const * p = parameters;
    
    // value

    REAL const arg1 = p[0] + p[1] * x - exp(x);
    REAL const arg2 = p[2] + p[3] * sin(x) - cos(x);

    value[point_index] = arg1*arg1 + arg2*arg2;

    // derivative

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = 2 * arg1;
    current_derivative[1 * n_points] = 2 * x * arg1;
    current_derivative[2 * n_points] = 2 * arg2;
    current_derivative[3 * n_points] = 2 * sin(x) * arg2;
}

#endif
