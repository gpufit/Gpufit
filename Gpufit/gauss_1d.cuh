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
* parameters: An input vector of concatenated sets of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate
*             p[2]: width (standard deviation)
*             p[3]: offset
*
* n_fits: The number of fits. (not used)
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
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The number of elements in user_info. (not used)
*
* Calling the calculate_gauss1d function
* ======================================
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

__device__ void calculate_gauss1d(
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
    float const * p = parameters;
    
    float const argx = (point_index - p[1]) * (point_index - p[1]) / (2 * p[2] * p[2]);
    float const ex = exp(-argx);
    values[point_index] = p[0] * ex + p[3];

    // derivatives

    float * current_derivative = derivatives + point_index;

    current_derivative[0]  = ex;
    current_derivative[1 * n_points]  = p[0] * ex * (point_index - p[1]) / (p[2] * p[2]);
    current_derivative[2 * n_points]  = p[0] * ex * (point_index - p[1]) * (point_index - p[1]) / (p[2] * p[2] * p[2]);
    current_derivative[3 * n_points]  = 1.f;
}

#endif
