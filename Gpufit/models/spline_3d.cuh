#ifndef GPUFIT_SPLINE3D_CUH_INCLUDED
#define GPUFIT_SPLINE3D_CUH_INCLUDED

/* Description of the calculate_spline3d function
* ===============================================
*
* This function calculates a value of a three-dimensional spline model function
* and its partial derivatives with respect to the model parameters.
*
* No independent variables are passed to this model function.  Hence, the
* (X, Y, Z) coordinate of the first data value is assumed to be (0.0, 0.0, 0.0). 
* For a fit size of M x N x O data points, the (X, Y, Z) coordinates of the data
* are simply the corresponding array index values of the data array, starting from
* zero.
*
* Parameters:
*
* parameters: An input vector of concatenated sets of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate x
*             p[2]: center coordinate y
*             p[3]: center coordinate z
*             p[4]: offset
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
*                user_info[2]: the number of data points in Z
*                user_info[3]: the number of spline intervals in X
*                user_info[4]: the number of spline intervals in Y
*                user_info[5]: the number of spline intervals in Z
*                user_info[6]: the value of coefficient (0,0,0) of interval (0,0,0)
*                user_info[7]: the value of coefficient (1,0,0) of interval (0,0,0)
*                   .
*                   .
*                   .
*                user_info[10]: the value of coefficient (0,1,0) of interval (0,0,0)
*                   .
*                   .
*                   .
*                user_info[17]: the value of coefficient (0,0,1) of interval (0,0,0)
*                   .
*                   .
*                   .
*                user_info[70]: the value of coefficient (0,0,0) of intervall (1,0,0)
*                   .
*                   .
*                   .
*
* user_info_size: The number of elements in user_info.
*
* Calling the calculate_spline3d function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_spline3d(
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
    std::size_t const n_points_y = static_cast<std::size_t>(*(user_info_REAL + 1));
    // std::size_t const n_points_z = static_cast<std::size_t>(*(user_info_REAL + 2)); // not needed
    int const n_intervals_x = static_cast<int>(*(user_info_REAL + 3));
    int const n_intervals_y = static_cast<int>(*(user_info_REAL + 4));
    int const n_intervals_z = static_cast<int>(*(user_info_REAL + 5));
    std::size_t const n_coefficients_per_interval = 64;
    REAL const * coefficients = user_info_REAL + 6;

    // parameters
    REAL const * p = parameters;

    // estimate coordinates (i, j, k) of the current interval
    int const point_index_x = point_index % n_points_x;
    int const point_index_y = (point_index / n_points_x) % n_points_y;
    int const point_index_z = point_index / (n_points_x * n_points_y);
    REAL const position_x = point_index_x - p[1];
    REAL const position_y = point_index_y - p[2];
    REAL const position_z = point_index_z - p[3];
    int i = static_cast<int>(floor(position_x));
    int j = static_cast<int>(floor(position_y));
    int k = static_cast<int>(floor(position_z));

    // adjust i, j and k to their bounds
    i = i >= 0 ? i : 0;
    i = i < n_intervals_x ? i : n_intervals_x - 1;
    j = j >= 0 ? j : 0;
    j = j < n_intervals_y ? j : n_intervals_y - 1;
    k = k >= 0 ? k : 0;
    k = k < n_intervals_z ? k : n_intervals_z - 1;

    // get coefficients of the current interval
    REAL const * current_coefficients
        = coefficients
        + (i * n_intervals_y * n_intervals_z + j * n_intervals_z + k)
        * n_coefficients_per_interval;

    // estimate position relative to the current spline interval
    REAL const x_diff = position_x - i;
    REAL const y_diff = position_y - j;
    REAL const z_diff = position_z - k;

    // intermediate values
    REAL temp_value = 0;
    REAL temp_derivative_1 = 0;
    REAL temp_derivative_2 = 0;
    REAL temp_derivative_3 = 0;

    REAL power_factor_i = 1;
    for (int order_i = 0; order_i < 4; order_i++)
    {
        REAL power_factor_j = 1;
        for (int order_j = 0; order_j < 4; order_j++)
        {
            REAL power_factor_k = 1;
            for (int order_k = 0; order_k < 4; order_k++)
            {
                // intermediate function value without amplitude and offset
                temp_value
                    += current_coefficients[order_i * 16 + order_j * 4 + order_k]
                    * power_factor_i
                    * power_factor_j
                    * power_factor_k;

                // intermediate derivative value with respect to paramater 1 (center position x)
                if (order_i < 3)
                {
                    temp_derivative_1
                        += (REAL(order_i) + 1)
                        * current_coefficients[(order_i + 1) * 16 + order_j * 4 + order_k]
                        * power_factor_i
                        * power_factor_j
                        * power_factor_k;
                }
                // intermediate derivative value with respect to paramater 2 (center position y)
                if (order_j < 3)
                {
                    temp_derivative_2
                        += (REAL(order_j) + 1)
                        * current_coefficients[order_i * 16 + (order_j + 1) * 4 + order_k]
                        * power_factor_i
                        * power_factor_j
                        * power_factor_k;
                }
                // intermediate derivative value with respect to paramater 3 (center position z)
                if (order_k < 3)
                {
                    temp_derivative_3
                        += (REAL(order_k) + 1)
                        * current_coefficients[order_i * 16 + order_j * 4 + (order_k + 1)]
                        * power_factor_i
                        * power_factor_j
                        * power_factor_k;
                }
                power_factor_k *= z_diff;
            }
            power_factor_j *= y_diff;
        }
        power_factor_i *= x_diff;
    }

    // value

    value[point_index] = p[0] * temp_value + p[4];

    // derivative

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = temp_value;
    current_derivative[1 * n_points] = -p[0] * temp_derivative_1;
    current_derivative[2 * n_points] = -p[0] * temp_derivative_2;
    current_derivative[3 * n_points] = -p[0] * temp_derivative_3;
    current_derivative[4 * n_points] = 1;
}

#endif
