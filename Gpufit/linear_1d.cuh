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
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    int const n_fits_per_block = blockDim.x / n_points;
    int const fit_in_block = threadIdx.x / n_points;
    int const point_index = threadIdx.x - (fit_in_block*n_points);
    int const fit_index = blockIdx.x * n_fits_per_block + fit_in_block;

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

    float* current_value = &values[fit_index*n_points];
    float const * current_parameters = &parameters[fit_index * n_parameters];

    current_value[point_index] = current_parameters[0] + current_parameters[1] * x;

    // derivatives

    float * current_derivative = &derivatives[fit_index * n_parameters * n_points + point_index];
    current_derivative[0] = 1.f;
    current_derivative[1 * n_points] = x;
}

#endif
