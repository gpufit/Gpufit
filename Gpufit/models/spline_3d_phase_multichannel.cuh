#ifndef GPUFIT_SPLINE3D_PHASE_MULTICHANNEL_CUH_INCLUDED
#define GPUFIT_SPLINE3D_PHASE_MULTICHANNEL_CUH_INCLUDED

/* Description of the calculate_spline3d_phase_multichannel function
* ==================================================================
*
* This function calculates a value of a combination of three three-dimensional splines
* with multiple channels and its partial derivatives with respect to the
* model parameters. The spline coefficients are obtained from data decomposed in
* three parts: data mean, modulation and modulation with phase shifted by 90 degree.
*
* No independent variables are passed to this model function.  Hence, the
* (X, Y, Z, CH) coordinate of the first data value is assumed to be (0.0, 0.0, 0.0, 0).
* For a fit size of M x N x O x P data points, the (X, Y, Z, CH) coordinates of the data
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
*             p[5]: phase
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
*                user_info[0]: the number of channels
*                user_info[1]: the number of data points in X
*                user_info[2]: the number of data points in Y
*                user_info[3]: the number of data points in Z
*                user_info[4]: the number of spline intervals in X
*                user_info[5]: the number of spline intervals in Y
*                user_info[6]: the number of spline intervals in Z
*                user_info[7]: the value of coefficient (0,0,0) of interval (0,0,0,0)
*                user_info[8]: the value of coefficient (1,0,0) of interval (0,0,0,0)
*                   .
*                   .
*                   .
*                user_info[11]: the value of coefficient (0,1,0) of interval (0,0,0,0)
*                   .
*                   .
*                   .
*                user_info[18]: the value of coefficient (0,0,1) of interval (0,0,0,0)
*                   .
*                   .
*                   .
*                user_info[71]: the value of coefficient (0,0,0) of intervall (1,0,0,0)
*                   .
*                   .
*                   .
*                user_info[7+Ni*64]: the value of coefficient (0,0,0) of intervall (0,0,0,1) of spline 0
*                    (Ni: the number of intervals per channel)
*                   .
*                   .
*                   .
*                user_info[7+Nc*Ni*64]: the value of coefficient (0,0,0) of intervall (0,0,0,0) of spline 1
*                    (Ni: the number of intervals per channel, Nc: the number of channels)
*                   .
*                   .
*                   .
*
* user_info_size: The number of elements in user_info.
*
* Calling the calculate_spline3d_phase_multichannel function
* ==========================================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_spline3d_phase_multichannel(
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

    std::size_t const n_channels = static_cast<std::size_t>(*(user_info_REAL + 0));
    std::size_t const n_points_x = static_cast<std::size_t>(*(user_info_REAL + 1));
    std::size_t const n_points_y = static_cast<std::size_t>(*(user_info_REAL + 2));
    std::size_t const n_points_z = static_cast<std::size_t>(*(user_info_REAL + 3));
    int const n_intervals_x = static_cast<int>(*(user_info_REAL + 4));
    int const n_intervals_y = static_cast<int>(*(user_info_REAL + 5));
    int const n_intervals_z = static_cast<int>(*(user_info_REAL + 6));

    std::size_t const n_points_per_channel = n_points / n_channels;
    std::size_t const n_intervals_per_channel = n_intervals_x * n_intervals_y * n_intervals_z;
    std::size_t const n_coefficients_per_interval = 64;
    REAL const * coefficients = user_info_REAL + 7;

    // parameters
    REAL const * p = parameters;

    // estimate coordinates (i, j, k) of the current interval
    int const point_index_x = point_index % n_points_x;
    int const point_index_y = (point_index / n_points_x) % n_points_y;
    int const point_index_z = (point_index / (n_points_x * n_points_y)) % n_points_z;

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
    int const channel_id = point_index / n_points_per_channel;
    std::size_t const n_coefficients_per_channel = n_coefficients_per_interval * n_intervals_per_channel;
    std::size_t const n_coefficients_per_spline = n_channels * n_coefficients_per_channel;
    std::size_t const channel_begin = channel_id * n_coefficients_per_channel;
    std::size_t const interval_begin = ((i * n_intervals_y + j) * n_intervals_z + k) * n_coefficients_per_interval;

    REAL const * current_coefficients = coefficients + channel_begin + interval_begin;

    REAL const * coefficients_mean = current_coefficients + 0 * n_coefficients_per_spline;
    REAL const * coefficients_modulation = current_coefficients + 1 * n_coefficients_per_spline;
    REAL const * coefficients_modulation_90deg = current_coefficients + 2 * n_coefficients_per_spline;

    // estimate position relative to the current spline interval
    REAL const x_diff = position_x - i;
    REAL const y_diff = position_y - j;
    REAL const z_diff = position_z - k;

    // intermediate values
    REAL mean = 0;
    REAL modulation = 0;
    REAL modulation_90deg = 0;
    REAL derivative_mean_p1 = 0;
    REAL derivative_mean_p2 = 0;
    REAL derivative_mean_p3 = 0;
    REAL derivative_modulation_p1 = 0;
    REAL derivative_modulation_p2 = 0;
    REAL derivative_modulation_p3 = 0;
    REAL derivative_modulation_90deg_p1 = 0;
    REAL derivative_modulation_90deg_p2 = 0;
    REAL derivative_modulation_90deg_p3 = 0;

    REAL power_factor_i = 1;
    for (int order_i = 0; order_i < 4; order_i++)
    {
        REAL power_factor_j = 1;
        for (int order_j = 0; order_j < 4; order_j++)
        {
            REAL power_factor_k = 1;
            for (int order_k = 0; order_k < 4; order_k++)
            {
                REAL const power_factor = power_factor_i * power_factor_j * power_factor_k;

                int const coefficient_index = order_i * 16 + order_j * 4 + order_k;

                // intermediate function values without amplitude and offset
                mean += coefficients_mean[coefficient_index] * power_factor;

                modulation += coefficients_modulation[coefficient_index] * power_factor;

                modulation_90deg += coefficients_modulation_90deg[coefficient_index] * power_factor;

                // intermediate derivative values with respect to paramater 1 (center position x)
                if (order_i < 3)
                {
                    int const order_i_plus_1 = order_i + 1;
                    int const coefficients_index_d1 = order_i_plus_1 * 16 + order_j * 4 + order_k;
                    REAL const REAL_order_i_plus_1 = static_cast<REAL>(order_i_plus_1);

                    derivative_mean_p1
                        += REAL_order_i_plus_1 * coefficients_mean[coefficients_index_d1] * power_factor;

                    derivative_modulation_p1
                        += REAL_order_i_plus_1 * coefficients_modulation[coefficients_index_d1] * power_factor;

                    derivative_modulation_90deg_p1
                        += REAL_order_i_plus_1 * coefficients_modulation_90deg[coefficients_index_d1] * power_factor;
                }
                // intermediate derivative values with respect to paramater 2 (center position y)
                if (order_j < 3)
                {
                    int const order_j_plus_1 = order_j + 1;
                    int const coefficients_index_d2 = order_i * 16 + order_j_plus_1 * 4 + order_k;
                    REAL const REAL_order_j_plus_1 = static_cast<REAL>(order_j_plus_1);

                    derivative_mean_p2
                        += REAL_order_j_plus_1 * coefficients_mean[coefficients_index_d2] * power_factor;

                    derivative_modulation_p2
                        += REAL_order_j_plus_1 * coefficients_modulation[coefficients_index_d2] * power_factor;

                    derivative_modulation_90deg_p2
                        += REAL_order_j_plus_1 * coefficients_modulation_90deg[coefficients_index_d2] * power_factor;
                }
                // intermediate derivative values with respect to paramater 3 (center position z)
                if (order_k < 3)
                {
                    int const order_k_plus_1 = order_k + 1;
                    int const coefficients_index_d3 = order_i * 16 + order_j * 4 + order_k_plus_1;
                    REAL const REAL_order_k_plus_1 = static_cast<REAL>(order_k_plus_1);

                    derivative_mean_p3
                        += REAL_order_k_plus_1 * coefficients_mean[coefficients_index_d3] * power_factor;

                    derivative_modulation_p3
                        += REAL_order_k_plus_1 * coefficients_modulation[coefficients_index_d3] * power_factor;

                    derivative_modulation_90deg_p3
                        += REAL_order_k_plus_1 * coefficients_modulation_90deg[coefficients_index_d3] * power_factor;
                }
                power_factor_k *= z_diff;
            }
            power_factor_j *= y_diff;
        }
        power_factor_i *= x_diff;
    }

    // reconstruct the function by recombining the mean, the modulation and  the
    // shifted modulation considering the phase parameter p[5]
    REAL const cos_phi = std::cos(p[5]);
    REAL const sin_phi = std::sin(p[5]);
    REAL const temp_value = mean + cos_phi * modulation + sin_phi * modulation_90deg;

    // value

    value[point_index] = p[0] * temp_value + p[4];

    // derivative

    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = temp_value;
    current_derivative[1 * n_points]
        = -p[0] * (derivative_mean_p1 + cos_phi * derivative_modulation_p1 + sin_phi * derivative_modulation_90deg_p1);
    current_derivative[2 * n_points]
        = -p[0] * (derivative_mean_p2 + cos_phi * derivative_modulation_p2 + sin_phi * derivative_modulation_90deg_p2);
    current_derivative[3 * n_points]
        = -p[0] * (derivative_mean_p3 + cos_phi * derivative_modulation_p3 + sin_phi * derivative_modulation_90deg_p3);
    current_derivative[4 * n_points] = 1;
    current_derivative[5 * n_points] = p[0] * (- sin_phi * modulation + cos_phi * modulation_90deg);
}

#endif
