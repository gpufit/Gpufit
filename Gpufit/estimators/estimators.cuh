#ifndef GPUFIT_ESTIMATORS_CUH_INCLUDED
#define GPUFIT_ESTIMATORS_CUH_INCLUDED

#include "lse.cuh"
#include "mle.cuh"

__device__ void calculate_chi_square(
    int const estimator_id,
    volatile REAL * chi_square,
    int const point_index,
    REAL const * data,
    REAL const * value,
    REAL const * weight,
    int * state,
    char * user_info,
    std::size_t const user_info_size)
{
    switch (estimator_id)
    {
    case LSE:
        calculate_chi_square_lse(chi_square, point_index, data, value, weight, state, user_info, user_info_size);
        break;
    case MLE:
        calculate_chi_square_mle(chi_square, point_index, data, value, weight, state, user_info, user_info_size);
        break;
    default:
        assert(0); // unknown estimator ID
    }
}

__device__ void calculate_gradient(
    int const estimator_id,
    volatile REAL * gradient,
    int const point_index,
    int const parameter_index,
    REAL const * data,
    REAL const * value,
    REAL const * derivative,
    REAL const * weight,
    char * user_info,
    std::size_t const user_info_size)
{
    switch (estimator_id)
    {
    case LSE:
        calculate_gradient_lse(gradient, point_index, parameter_index, data, value, derivative, weight, user_info, user_info_size);
        break;
    case MLE:
        calculate_gradient_mle(gradient, point_index, parameter_index, data, value, derivative, weight, user_info, user_info_size);
        break;
    default:
        assert(0); // unknown estimator ID
    }
}

__device__ void calculate_hessian(
    int const estimator_id,
    double * hessian,
    int const point_index,
    int const parameter_index_i,
    int const parameter_index_j,
    REAL const * data,
    REAL const * value,
    REAL const * derivative,
    REAL const * weight,
    char * user_info,
    std::size_t const user_info_size)
{
    switch (estimator_id)
    {
    case LSE:
        calculate_hessian_lse
        (hessian, point_index, parameter_index_i, parameter_index_j, data, value, derivative, weight, user_info,user_info_size);
        break;
    case MLE:
        calculate_hessian_mle
        (hessian, point_index, parameter_index_i, parameter_index_j, data, value, derivative, weight, user_info, user_info_size);
        break;
    default:
        assert(0); // unknown estimator ID
    }
}

#endif