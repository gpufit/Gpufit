#ifndef GPUFIT_SPLINE1D_CUH_INCLUDED
#define GPUFIT_SPLINE1D_CUH_INCLUDED

/* Description of the calculate_spline1d function
* ===============================================
*
* This function calculates a value of a one-dimensional cubic spline model function
* and its partial derivatives with respect to the model parameters. 
*
* No independent variables are passed to this model function.  Hence, the
* X coordinate of the first data value is assumed to be 0.0.  For
* a fit size of N data points, the X coordinates of the data are
* simply the corresponding array index values of the data array, starting from
* zero.
*
* Parameters:
*
* parameters: An input vector of concatenated sets of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate
*             p[2]: offset
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
*                user_info[1]: the number of spline intervals in X
*                user_info[2]: the value of coefficient 0 of interval 0
*                user_info[3]: the value of coefficient 1 of interval 0
*                   .
*                   .
*                   .
*                user_info[6]: the value of coefficient 0 of interval 1
*                   .
*                   .
*                   .
*
* user_info_size: The number of elements in user_info.
*
* Calling the calculate_spline1d function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_spline1d(
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

    int const n_intervals = static_cast<int>(*user_info_REAL);
    std::size_t const n_coefficients_per_interval = 4;

    REAL const * coefficients = user_info_REAL + 1;

    // parameters
    REAL const * p = parameters;

    // estimate index i of the current spline interval
    REAL const x = static_cast<REAL>(point_index);
    REAL const position = x - p[1];
    int i = static_cast<int>(floor(position)); // can be negative
    
    // adjust i to its bounds
    i = i >= 0 ? i : 0;
    i = i < n_intervals ? i : n_intervals - 1;

    // get coefficients of the current interval
    REAL const * current_coefficients = coefficients + i * n_coefficients_per_interval;

    // calculate position relative to the current spline interval
    REAL const x_diff = position - static_cast<REAL>(i);

    // intermediate values
    REAL temp_value = 0;
    REAL temp_derivative_1 = 0;

    REAL power_factor = 1;
    for (std::size_t order = 0; order < n_coefficients_per_interval; order++)
    {
        // intermediate function value without amplitude and offset
        temp_value += current_coefficients[order] * power_factor;

        // intermediate derivative value with respect to paramater 1 (center position)
        if (order < n_coefficients_per_interval - 1)
            temp_derivative_1
                += (REAL(order) + 1)
                * current_coefficients[order + 1]
                * power_factor;
            
        power_factor *= x_diff;
    }

    // value
    value[point_index] = p[0] * temp_value + p[2];

    // derivative
    REAL * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = temp_value;
    current_derivative[1 * n_points] = -p[0] * temp_derivative_1;
    current_derivative[2 * n_points] = 1;
}

#endif
