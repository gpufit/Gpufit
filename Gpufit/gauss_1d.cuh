#ifndef GPUFIT_GAUSS1D_CUH_INCLUDED
#define GPUFIT_GAUSS1D_CUH_INCLUDED

/* Description of the calculate_gauss1d function
* ==============================================
*
* This function calculates the values of one-dimensional gauss model functions
* and their partial derivatives with respect to the model parameters. 
*
* No independent variables are passed to this model function.  Hence, the 
* (X) coordinate of the first data value is assumed to be (0.0).  For
* a fit size of M data points, the (X) coordinates of the data are
* simply the corresponding array index values of the data array, starting from
* zero.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate
*             p[2]: width (standard deviation)
*             p[3]: offset
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
* Calling the calculate_gauss1d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_gauss1d(
    float const * parameters,
    int const n_fits,
    int const n_points,
    float * value,
    float * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // parameters

    float const * p = parameters;
    
    // value

    float const argx = (point_index - p[1]) * (point_index - p[1]) / (2 * p[2] * p[2]);
    float const ex = exp(-argx);
    value[point_index] = p[0] * ex + p[3];

    // derivative

    float * current_derivative = derivative + point_index;

    current_derivative[0 * n_points]  = ex;
    current_derivative[1 * n_points]  = p[0] * ex * (point_index - p[1]) / (p[2] * p[2]);
    current_derivative[2 * n_points]  = p[0] * ex * (point_index - p[1]) * (point_index - p[1]) / (p[2] * p[2] * p[2]);
    current_derivative[3 * n_points]  = 1.f;
}

#endif
