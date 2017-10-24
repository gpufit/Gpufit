#include "lse.cuh"
#include "mle.cuh"

__device__ void(*calculate_chi_square) (
    volatile float * chi_square,
    int const point_index,
    float const * data,
    float const * value,
    float const * weight,
    int * state,
    char * user_info,
    std::size_t const user_info_size);

__device__ void(*calculate_gradient) (
    volatile float * gradient,
    int const point_index,
    int const parameter_index,
    float const * data,
    float const * value,
    float const * derivative,
    float const * weight,
    char * user_info,
    std::size_t const user_info_size);

__device__ void(*calculate_hessian) (
    double * hessian,
    int const point_index,
    int const parameter_index_i,
    int const parameter_index_j,
    float const * data,
    float const * value,
    float const * derivative,
    float const * weight,
    char * user_info,
    std::size_t const user_info_size);

__device__ void device_configure_estimator(int const estimator_id)
{
    switch (estimator_id)
    {
    case 0: calculate_chi_square = calculate_chi_square_lse;
            calculate_gradient   = calculate_gradient_lse;
            calculate_hessian    = calculate_hessian_lse;
        break;
            
    case 1: calculate_chi_square = calculate_chi_square_mle;
            calculate_gradient   = calculate_gradient_mle;
            calculate_hessian    = calculate_hessian_mle;
        break;

    default:
        break;
    }
}