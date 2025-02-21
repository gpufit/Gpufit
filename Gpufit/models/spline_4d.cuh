/* Description of the calculate_spline4d function
* ===============================================
*
* This function calculates a value of a five-dimensional spline model function
* and its partial derivatives with respect to the model parameters.
*
* No independent variables are passed to this model function.  Hence, the
* (X, Y, Z, W) coordinate of the first data value is assumed to be (0, 0, 0, 0). 
* For a fit size of M x N x O x P data points, the (X, Y, Z, W) coordinates of 
* the data are simply the corresponding array index values of the data array, 
* starting from zero.
*
* Parameters:
*
* parameters: An input vector of concatenated sets of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate x
*             p[2]: center coordinate y
*             p[3]: center coordinate z
*             p[4]: center coordinate w
*             p[5]: offset
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
*                user_info[3]: the number of data points in W
*                user_info[4]: the number of spline intervals in X
*                user_info[5]: the number of spline intervals in Y
*                user_info[6]: the number of spline intervals in Z
*                user_info[7]: the number of spline intervals in W
*                user_info[8]: the value of coefficient (0,0,0,0) of interval (0,0,0,0)
*                user_info[9]: the value of coefficient (1,0,0,0) of interval (0,0,0,0)
*                   .
*                   .
*                   .
*                user_info[xx]: the value of coefficient (0,1,0,0) of interval (0,0,0,0)
*                   .
*                   .
*                   .
*                user_info[xx]: the value of coefficient (0,0,1,0) of interval (0,0,0,0)
*                   .
*                   .
*                   .
*                user_info[xx]: the value of coefficient (0,0,0,0) of intervall (1,0,0,0)
*                   .
*                   .
*                   .
*
* user_info_size: The number of elements in user_info.
*
* Calling the calculate_spline4d function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_spline4d(
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
    // Read user_info (cast from char* to REAL*)
    REAL const * user_info_REAL = (REAL *)user_info;

    // Read data dimensions and spline intervals
    std::size_t const n_points_x = static_cast<std::size_t>(*(user_info_REAL + 0));
    std::size_t const n_points_y = static_cast<std::size_t>(*(user_info_REAL + 1));
    std::size_t const n_points_z = static_cast<std::size_t>(*(user_info_REAL + 2));
    std::size_t const n_points_w = static_cast<std::size_t>(*(user_info_REAL + 3));

    int const n_intervals_x = static_cast<int>(*(user_info_REAL + 4));
    int const n_intervals_y = static_cast<int>(*(user_info_REAL + 5));
    int const n_intervals_z = static_cast<int>(*(user_info_REAL + 6));
    int const n_intervals_w = static_cast<int>(*(user_info_REAL + 7));

    std::size_t const n_coefficients_per_interval = 256; // 4^4 coefficients per interval
    REAL const * coefficients = user_info_REAL + 8; // Start of coefficient storage

    // Model parameters
    REAL const * p = parameters;

    // Compute the (x, y, z, w) coordinate of the current data point
    int const point_index_x = point_index % n_points_x;
    int const point_index_y = (point_index / n_points_x) % n_points_y;
    int const point_index_z = (point_index / (n_points_x * n_points_y)) % n_points_z;
    int const point_index_w = point_index / (n_points_x * n_points_y * n_points_z);

    // Compute position relative to the spline center
    REAL const position_x = point_index_x - p[1];
    REAL const position_y = point_index_y - p[2];
    REAL const position_z = point_index_z - p[3];
    REAL const position_w = point_index_w - p[4];

    // Determine the spline interval indices (i, j, k, m)
    int i = static_cast<int>(floor(position_x));
    int j = static_cast<int>(floor(position_y));
    int k = static_cast<int>(floor(position_z));
    int m = static_cast<int>(floor(position_w));

    // Clamp indices to valid bounds
    i = max(0, min(i, n_intervals_x - 1));
    j = max(0, min(j, n_intervals_y - 1));
    k = max(0, min(k, n_intervals_z - 1));
    m = max(0, min(m, n_intervals_w - 1));

    // Get the correct coefficient set for this interval
    REAL const * current_coefficients
        = coefficients
        + ((i * n_intervals_y * n_intervals_z * n_intervals_w) +
           (j * n_intervals_z * n_intervals_w) +
           (k * n_intervals_w) + m) * n_coefficients_per_interval;

    // Compute fractional positions within the current interval
    REAL const x_diff = position_x - i;
    REAL const y_diff = position_y - j;
    REAL const z_diff = position_z - k;
    REAL const w_diff = position_w - m;

    // Intermediate values for function and derivatives
    REAL temp_value = 0;
    REAL temp_derivative_1 = 0;
    REAL temp_derivative_2 = 0;
    REAL temp_derivative_3 = 0;
    REAL temp_derivative_4 = 0;

    REAL power_factor_i = 1;
    for (int order_i = 0; order_i < 4; order_i++) {
        REAL power_factor_j = 1;
        for (int order_j = 0; order_j < 4; order_j++) {
            REAL power_factor_k = 1;
            for (int order_k = 0; order_k < 4; order_k++) {
                REAL power_factor_m = 1;
                for (int order_m = 0; order_m < 4; order_m++) {

                    // Compute function value without amplitude & offset
                    temp_value += current_coefficients[order_i * 64 + order_j * 16 + order_k * 4 + order_m]
                                  * power_factor_i * power_factor_j * power_factor_k * power_factor_m;

                    // Compute derivative w.r.t. x (center position)
                    if (order_i < 3) {
                        temp_derivative_1 += (REAL(order_i) + 1)
                                             * current_coefficients[(order_i + 1) * 64 + order_j * 16 + order_k * 4 + order_m]
                                             * power_factor_i * power_factor_j * power_factor_k * power_factor_m;
                    }

                    // Compute derivative w.r.t. y
                    if (order_j < 3) {
                        temp_derivative_2 += (REAL(order_j) + 1)
                                             * current_coefficients[order_i * 64 + (order_j + 1) * 16 + order_k * 4 + order_m]
                                             * power_factor_i * power_factor_j * power_factor_k * power_factor_m;
                    }

                    // Compute derivative w.r.t. z
                    if (order_k < 3) {
                        temp_derivative_3 += (REAL(order_k) + 1)
                                             * current_coefficients[order_i * 64 + order_j * 16 + (order_k + 1) * 4 + order_m]
                                             * power_factor_i * power_factor_j * power_factor_k * power_factor_m;
                    }

                    // Compute derivative w.r.t. w (NEW)
                    if (order_m < 3) {
                        temp_derivative_4 += (REAL(order_m) + 1)
                                             * current_coefficients[order_i * 64 + order_j * 16 + order_k * 4 + (order_m + 1)]
                                             * power_factor_i * power_factor_j * power_factor_k * power_factor_m;
                    }

                    power_factor_m *= w_diff;
                }
                power_factor_k *= z_diff;
            }
            power_factor_j *= y_diff;
        }
        power_factor_i *= x_diff;
    }

    // Compute final function value
    value[point_index] = p[0] * temp_value + p[5];

    // Store derivatives
    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = temp_value;
    current_derivative[1 * n_points] = -p[0] * temp_derivative_1;
    current_derivative[2 * n_points] = -p[0] * temp_derivative_2;
    current_derivative[3 * n_points] = -p[0] * temp_derivative_3;
    current_derivative[4 * n_points] = -p[0] * temp_derivative_4;
    current_derivative[5 * n_points] = 1;
}
