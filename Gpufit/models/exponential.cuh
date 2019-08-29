#ifdef USE_BASE_MODELS
#define GPUFIT_EXPONENTIAL_CUH_INCLUDED

//TODO: start putting the actual exponential function together

__device__ void  calculate_exponential(
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

    REAL * user_info_float = (REAL*) user_info;
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

	///////////////////////////// value //////////////////////////////

    value[point_index] = parameters[1] * exp(-1 * parameters[0] * x) ;                      // formula calculating fit model values

    /////////////////////////// derivative ///////////////////////////
    float * current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = -1 * parameters[0] * parameters[1] * exp(-1 * parameters[0] * x) ;  // formula calculating derivative values with respect to parameters[0]
    current_derivative[1 * n_points] = exp(-1 * parameters[0] * x) ;  // formula calculating derivative values with respect to parameters[1]

}

#endif
