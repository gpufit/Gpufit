#ifndef GPUFIT_FLETCHERPOWELLHELIX_CUH_INCLUDED
#define GPUFIT_FLETCHERPOWELLHELIX_CUH_INCLUDED

/* Description of the calculate_fletcher_powell_helix function
* ============================================================
*
* This function calculates the values of one-dimensional helix function defined
* in [Fletcher, R. & Powell, M. J. D. A rapidly convergent descent method for
* minimization. The Computer Journal 6 (1963)] and their partial derivatives
* with respect to the model parameters.
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
* Calling the calculate_fletcher_powell_helix function
* ====================================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_fletcher_powell_helix(
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
    // parameters

    REAL const * p = parameters;
    
    // arguments

    REAL const pi = 3.14159f;

    REAL theta = 0;

    if (p[0] > 0)
        theta = .5f * atan(p[1] / p[0]) / pi;
    else if (p[0] < 0)
        theta = .5f * atan(p[1] / p[0]) / pi + .5f;

    REAL const arg = p[0] * p[0] + p[1] * p[1];

    // values and derivatives

    switch (point_index)
    {
    case 0:
        // value
        value[point_index] = 10 * (p[2] - 10 * theta);
        // derivative
        derivative[0 * n_points + point_index] = 100 / (2*pi) * p[1] / arg;
        derivative[1 * n_points + point_index] = -100 / (2*pi) * p[0] / arg;
        derivative[2 * n_points + point_index] = 10;
        break;
    case 1:
        // value
        value[point_index] = 10 * (std::sqrt(arg) - 1);
        // derivative
        derivative[0 * n_points + point_index] = 10 * p[0] / std::sqrt(arg);
        derivative[1 * n_points + point_index] = 10 * p[1] / std::sqrt(arg);
        derivative[2 * n_points + point_index] = 0;
        break;
    case 2:
        // value
        value[point_index] = p[2];
        // derivative
        derivative[0 * n_points + point_index] = 0;
        derivative[1 * n_points + point_index] = 0;
        derivative[2 * n_points + point_index] = 1;
        break;
    default:
        break;
    }

    
}

#endif
