#include "gauss_1d.cuh"
#include "gauss_2d.cuh"
#include "gauss_2d_elliptic.cuh"
#include "gauss_2d_rotated.cuh"
#include "cauchy_2d_elliptic.cuh"
#include "linear_1d.cuh"

__device__ void(*calculate_model) (
    float const * parameters,
    int const n_fits,
    int const n_points,
    float * value,
    float * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size);

__device__ void device_configure_model(int const model_id, int * n_parameters, int * n_dimensions)
{
    switch (model_id)
    {
    case 0: calculate_model = calculate_gauss1d;
            *n_parameters = 4;
            *n_dimensions = 1;                               break;
    case 1: calculate_model = calculate_gauss2d;
            *n_parameters = 5;
            *n_dimensions = 2;                               break;
    case 2: calculate_model = calculate_gauss2delliptic;
            *n_parameters = 6;
            *n_dimensions = 2;                               break;
    case 3: calculate_model = calculate_gauss2drotated;
            *n_parameters = 7;
            *n_dimensions = 2;                               break;
    case 4: calculate_model = calculate_cauchy2delliptic;
            *n_parameters = 6;
            *n_dimensions = 2;                               break;
    case 5: calculate_model = calculate_linear1d;
            *n_parameters = 2;
            *n_dimensions = 1;                               break;
    default:                                                break;
    }
}