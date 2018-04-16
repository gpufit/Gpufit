#ifndef GPUFIT_GAUSS2D_CUH_INCLUDED
#define GPUFIT_GAUSS2D_CUH_INCLUDED

/* Description of the calculate_gauss2d function
* ==============================================
*
* This function calculates the values of two-dimensional gauss model functions
* and their partial derivatives with respect to the model parameters. 
*
* No independent variables are passed to this model function.  Hence, the 
* (X, Y) coordinate of the first data value is assumed to be (0.0, 0.0).  For
* a fit size of M x N data points, the (X, Y) coordinates of the data are
* simply the corresponding array index values of the data array, starting from
* zero.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate x
*             p[2]: center coordinate y
*             p[3]: width (standard deviation; equal width in x and y dimensions)
*             p[4]: offset
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
* Calling the calculate_gauss2d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_gauss2d(
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

    int const n_points_x = sqrt((REAL)n_points);
    int const point_index_y = point_index / n_points_x;
    int const point_index_x = point_index - point_index_y * n_points_x;

    // parameters

    REAL const * p = parameters;

    // value

    REAL const argx = (point_index_x - p[1]) * (point_index_x - p[1]) / (2 * p[3] * p[3]);
    REAL const argy = (point_index_y - p[2]) * (point_index_y - p[2]) / (2 * p[3] * p[3]);
    REAL const ex = exp(-(argx + argy));
    value[point_index] = p[0] * ex + p[4];

    // derivatives

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = ex;
    current_derivative[1 * n_points] = p[0] * ex * (point_index_x - p[1]) / (p[3] * p[3]);
    current_derivative[2 * n_points] = p[0] * ex * (point_index_y - p[2]) / (p[3] * p[3]);
    current_derivative[3 * n_points] = ex * p[0] * ((point_index_x - p[1]) * (point_index_x - p[1]) + (point_index_y - p[2]) * (point_index_y - p[2])) / (p[3] * p[3] * p[3]);
    current_derivative[4 * n_points] = 1;
}

#endif
