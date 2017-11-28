#ifndef GPUFIT_DUALEXP_CUH_INCLUDED
#define GPUFIT_DUALEXP_CUH_INCLUDED

/* Description of the DUALEXP function
* ==============================================
*
* This function calculates the values of one-dimensional gauss model functions
* and their partial derivatives with respect to the model parameters. 
*
* This function makes use of the user information data to pass in the 
* independent variables (X values) corresponding to the data.  The X values
* must be of type float.
*
* There are three possibilities regarding the X values:
*
*   No X values provided: 
*
*       If no user information is provided, the (X) coordinate of the 
*       first data value is assumed to be (0.0).  In this case, for a 
*       fit size of M data points, the (X) coordinates of the data are 
*       simply the corresponding array index values of the data array, 
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then 
*       the same X values will be used for all fits.  In this case, the 
*       size of the user_info array (in bytes) must equal 
*       sizeof(float) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array 
*       (in bytes) must equal sizeof(float) * n_points * nfits.
*
* Parameters:
* y = a*e^bx+c*e^dx;
* parameters: An input vector of model parameters.
*             p[0]: a 
*             p[1]: b
*             p[2]: c
*             p[3]: d
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
* user_info: An input vector containing user information. 
*
* user_info_size: The size of user_info in bytes. 
*
* Calling the calculate_gauss1d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_dualExp(
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
    // indices

    float * user_info_float = (float*)user_info;
    float x = 0.0f;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(float) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(float) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // parameters

    float const * p = parameters;
    
    // value

    //float const argx = (x - p[1]) * (x - p[1]) / (2 * p[2] * p[2]);
    //float const ex = exp(-argx);
    value[point_index] = p[0] * exp(p[1]*x) + p[2]*exp(p[3]*x);

    // derivative

    float * current_derivative = derivative + point_index;

    current_derivative[0 * n_points]  = exp(p[1]*x);
    current_derivative[1 * n_points]  = p[0] * exp(p[1]*x)*x;
    current_derivative[2 * n_points]  = exp(p[3]*x);
    current_derivative[3 * n_points]  = p[2] * exp(p[3]*x)*x;
}

#endif
