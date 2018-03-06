#ifndef GPUFIT_INTERFACE_H_INCLUDED
#define GPUFIT_INTERFACE_H_INCLUDED

#include "lm_fit.h"

static_assert( sizeof( int ) == 4, "32 bit 'int' type required" ) ;

class FitInterface
{
public:
    FitInterface
    (
        float const * data,
        float const * weights,
        std::size_t n_fits,
        int n_points,
        float tolerance,
        int max_n_iterations,
        EstimatorID estimator_id,
        float const * initial_parameters,
        int * parameters_to_fit,
        char * user_info,
        std::size_t user_info_size,
        float * output_parameters,
        int * output_states,
        float * output_chi_squares,
        int * output_n_iterations
    ) ;
    
    virtual ~FitInterface();
    void fit(ModelID const model_id);

private:
    void check_sizes();
    void configure_info(Info & info, ModelID const model_id);

public:

private:
    //input
    float const * const data_ ;
    float const * const weights_;
    float const * const initial_parameters_;
    int const * const parameters_to_fit_;
    char * const user_info_;
    int n_parameters_;

    std::size_t const n_fits_;
    int const n_points_;
    float const  tolerance_;
    int const max_n_iterations_;
    EstimatorID estimator_id_;
    std::size_t const user_info_size_;

    //output
    float * output_parameters_;
    int * output_states_;
    float * output_chi_squares_;
    int * output_n_iterations_;
};

#endif
