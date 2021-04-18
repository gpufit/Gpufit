#ifndef GPUFIT_SPLINE2D_CUH_INCLUDED
#define GPUFIT_SPLINE2D_CUH_INCLUDED

/* Description of the calculate_spline2d function
* ===============================================
*
* This function calculates a value of a two-dimensional spline model function and
* its partial derivatives with respect to the model parameters.
*
* No independent variables are passed to this model function.  Hence, the
* (X, Y) coordinate of the first data value is assumed to be (0.0, 0.0).  For
* a fit size of M x N data points, the (X, Y) coordinates of the data are
* simply the corresponding array index values of the data array, starting from
* zero.
*
* Parameters:
*
* parameters: An input vector of concatenated sets of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate x
*             p[2]: center coordinate y
*             p[3]: offset
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* values: An output vector of concatenated sets of model function values.
*
* derivatives: An output vector of concatenated sets of model function partial
*              derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index.
*
* chunk_index: The chunk index.
*
* user_info: An input vector containing user information. Here it is used to pass
*            in the data dimensions, spline dimensions and spline coefficients in
*            the following order:
*                user_info[0]: the number of data points in X
*                user_info[1]: the number of data points in Y
*                user_info[2]: the number of spline intervals in X
*                user_info[3]: the number of spline intervals in Y
*                user_info[4]: the value of coefficient (0,0) of interval (0,0)
*                user_info[5]: the value of coefficient (1,0) of interval (0,0)
*                   .
*                   .
*                   .
*                user_info[8]: the value of coefficient (0,1) of intervall (0,0)
*                   .
*                   .
*                   .
*                user_info[20]: the value of coefficient (0,0) of intervall (1,0)
*                   .
*                   .
*                   .
*
* user_info_size: The number of elements in user_info.
*
* Calling the calculate_spline2d function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_spline2d(
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
    // read user_info
    REAL const * user_info_REAL = (REAL *)user_info;

    std::size_t const n_points_x = static_cast<std::size_t>(*(user_info_REAL + 0));
    // std::size_t const n_points_y = static_cast<std::size_t>(*(user_info_REAL + 1)); // not needed
    int const n_intervals_x = static_cast<int>(*(user_info_REAL + 2));
    int const n_intervals_y = static_cast<int>(*(user_info_REAL + 3));

    std::size_t const n_coefficients_per_interval = 16;

    REAL const * coefficients = user_info_REAL + 4;

    // parameters
    REAL const * p = parameters;

    // estimate coordinates (i, j) of the current interval
    int const point_index_x = point_index % n_points_x;
    int const point_index_y = point_index / n_points_x;
    REAL const x = static_cast<REAL>(point_index_x);
    REAL const y = static_cast<REAL>(point_index_y);
    REAL const position_x = x - p[1];
    REAL const position_y = y - p[2];
    int i = static_cast<int>(floor(position_x));
    int j = static_cast<int>(floor(position_y));

    // adjust i and j to their bounds
    i = i >= 0 ? i : 0;
    i = i < n_intervals_x ? i : n_intervals_x - 1;
    j = j >= 0 ? j : 0;
    j = j < n_intervals_y ? j : n_intervals_y - 1;

    // get coefficients of the current interval
    REAL const * current_coefficients
        = coefficients + (i * n_intervals_y + j) * n_coefficients_per_interval;

    // estimate position relative to the current spline interval
    REAL const x_diff = position_x - static_cast<REAL>(i);
    REAL const y_diff = position_y - static_cast<REAL>(j);

    // intermediate values
    REAL temp_value = 0;
    REAL temp_derivative_1 = 0;
    REAL temp_derivative_2 = 0;

    REAL power_factor_i = 1;
    for (int order_i = 0; order_i < 4; order_i++)
    {
        REAL power_factor_j = 1;
        for (int order_j = 0; order_j < 4; order_j++)
        {
            // intermediate function value without amplitude and offset
            temp_value
                += current_coefficients[order_i * 4 + order_j]
                * power_factor_i
                * power_factor_j;

            // intermediate derivative value with respect to paramater 1 (center position x)
            if (order_i < 3)
            {
                temp_derivative_1
                    += (REAL(order_i) + 1)
                    * current_coefficients[(order_i + 1) * 4 + order_j]
                    * power_factor_i
                    * power_factor_j;
            }
            // intermediate derivative value with respect to paramater 2 (center position y)
            if (order_j < 3)
            {
                temp_derivative_2
                    += (REAL(order_j) + 1)
                    * current_coefficients[order_i * 4 + (order_j + 1)]
                    * power_factor_i
                    * power_factor_j;
            }

            power_factor_j *= y_diff;
        }
        power_factor_i *= x_diff;
    }
    // value

    value[point_index] = p[0] * temp_value + p[3];

    // derivative

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = temp_value;
    current_derivative[1 * n_points] = -p[0] * temp_derivative_1;
    current_derivative[2 * n_points] = -p[0] * temp_derivative_2;
    current_derivative[3 * n_points] = 1;
}

#endif
