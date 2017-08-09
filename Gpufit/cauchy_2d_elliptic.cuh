#ifndef GPUFIT_CAUCHY2DELLIPTIC_CUH_INCLUDED
#define GPUFIT_CAUCHY2DELLIPTIC_CUH_INCLUDED

/* Description of the calculate_cauchy2delliptic function
* =======================================================
*
* This function calculates the values of two-dimensional elliptic cauchy model
* functions and their partial derivatives with respect to the model parameters.
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
*             p[3]: width x (standard deviation)
*             p[4]: width y (standard deviation)
*             p[5]: offset
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
* Calling the calculate_cauchy2delliptic function
* ===============================================
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

__device__ void calculate_cauchy2delliptic(
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
    int const n_points_x = sqrt((float)n_points);
    int const n_fits_per_block = blockDim.x / n_points;
    int const fit_in_block = threadIdx.x / n_points;
    int const point_index = threadIdx.x - (fit_in_block*n_points);
    int const fit_index = blockIdx.x*n_fits_per_block + fit_in_block;

    int const point_index_y = point_index / n_points_x;
    int const point_index_x = point_index - (point_index_y*n_points_x);

    float* current_value = &values[fit_index*n_points];
    float const * p = &parameters[fit_index*n_parameters];
    
    float const argx  = ((p[1] - point_index_x) / p[3]) *((p[1] - point_index_x) / p[3]) + 1;
    float const argy = ((p[2] - point_index_y) / p[4]) *((p[2] - point_index_y) / p[4]) + 1;
    current_value[point_index] = p[0] * 1 / argx * 1 / argy + p[5];

    //////////////////////////////////////////////////////////////////////////////

    float * current_derivative = &derivatives[fit_index * n_points*n_parameters];

    current_derivative[0 * n_points + point_index]
        = 1 / (argx*argy);
    current_derivative[1 * n_points + point_index]
        = -2 * p[0] * (p[1] - point_index_x) * 1 / (p[3] * p[3] * argx*argx*argy);
    current_derivative[2 * n_points + point_index]
        = -2 * p[0] * (p[2] - point_index_y) * 1 / (p[4] * p[4] * argy*argy*argx);
    current_derivative[3 * n_points + point_index]
        = 2 * p[0] * (p[1] - point_index_x) * (p[1] - point_index_x)
        / (p[3] * p[3] * p[3] * argx * argx * argy);
    current_derivative[4 * n_points + point_index]
        = 2 * p[0] * (p[2] - point_index_y) * (p[2] - point_index_y) 
        / (p[4] * p[4] * p[4] * argy * argy * argx);
    current_derivative[5 * n_points + point_index]
        = 1;
}

#endif
