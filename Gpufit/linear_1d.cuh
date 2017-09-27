#ifndef GPUFIT_LINEAR1D_CUH_INCLUDED
#define GPUFIT_LINEAR1D_CUH_INCLUDED

/* Description of the calculate_linear1d function
* ===================================================
*
* This function calculates the values of one-dimensional linear model functions
* and their partial derivatives with respect to the model parameters. 
*
* This function makes use of the user information data to pass in the 
* independent variables (X values) corresponding to the data.  
*
* Note that if no user information is provided, the (X) coordinate of the 
* first data value is assumed to be (0.0).  In this case, for a fit size of 
* M data points, the (X) coordinates of the data are simply the corresponding 
* array index values of the data array, starting from zero.
*
* Parameters:
*
* parameters: An input vector of concatenated sets of model parameters.
*             p[0]: offset
*             p[1]: slope
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* n_parameters: The number of model parameters.
*
* values: An output vector of concatenated sets of model function values.
*
* derivatives: An output vector of concatenated sets of model function partial
*              derivatives.
*
* chunk_index: The chunk index. Used for indexing of user_info.
*
* user_info: An input vector containing user information.
*
* user_info_size: The number of elements in user_info.
*
* Calling the calculate_linear1d function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function. When calling the function, the blocks and threads of the __global__
* function must be set up correctly, as shown in the following example code.
*
*   dim3  threads(1, 1, 1);
*   dim3  blocks(1, 1, 1);
*
*   threads.x = n_points * n_fits_per_block;
*   blocks.x = n_fits / n_fits_per_block;
*
*   global_function<<< blocks,threads >>>(parameter1, ...);
*
*/

__device__ void calculate_linear1d(
    float const * parameters,
    int const n_fits,
    int const n_points,
    int const n_parameters,
    float * values,
    float * derivatives,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    float * user_info_float = (float*) user_info;
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

    values[point_index] = parameters[0] + parameters[1] * x;

    // derivatives

    float * current_derivatives = derivatives + point_index;
    current_derivatives[0 * n_points] = 1.f;
    current_derivatives[1 * n_points] = x;
}

#endif
