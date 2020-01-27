#ifndef GPUFIT_INTERFACE_H_INCLUDED
#define GPUFIT_INTERFACE_H_INCLUDED

#include "lm_fit.h"

static_assert( sizeof( int ) == 4, "32 bit 'int' type required" ) ;

class FitInterface
{
public:
    FitInterface
    (
        REAL const * data,
        REAL const * weights,
        std::size_t n_fits,
        int n_points,
        REAL tolerance,
        int max_n_iterations,
        EstimatorID estimator_id,
        REAL const * initial_parameters,
        int * parameters_to_fit,
        REAL const * constraints,
        int const * constraint_types,
        char * user_info,
        std::size_t user_info_size,
        REAL * output_parameters,
        int * output_states,
        REAL * output_chi_squares,
        int * output_n_iterations,
        DataLocation data_location
    ) ;
    
    virtual ~FitInterface();
    void fit(ModelID const model_id);

private:
    void check_sizes();
    void configure_info(Info & info, ModelID const model_id);

public:

private:
    //input
    REAL const * const data_ ;
    REAL const * const weights_;
    REAL const * const initial_parameters_;
    int const * const parameters_to_fit_;
    REAL const * const constraints_;
    int const * const constraint_types_;
    char * const user_info_;
    int n_parameters_;

    std::size_t const n_fits_;
    int const n_points_;
    REAL const  tolerance_;
    int const max_n_iterations_;
    EstimatorID estimator_id_;
    std::size_t const user_info_size_;

    DataLocation data_location_;

    //output
    REAL * output_parameters_;
    int * output_states_;
    REAL * output_chi_squares_;
    int * output_n_iterations_;
};

#endif
