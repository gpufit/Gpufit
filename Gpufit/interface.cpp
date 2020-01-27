#include "gpufit.h"
#include "interface.h"
#include "cuda_kernels.cuh"

FitInterface::FitInterface
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
) :
    data_( data ),
    weights_( weights ),
    initial_parameters_( initial_parameters ),
    parameters_to_fit_( parameters_to_fit ),
    constraints_(constraints),
    constraint_types_(constraint_types),
    user_info_( user_info ),
    n_fits_(n_fits),
    n_points_(n_points),
    tolerance_(tolerance),
    max_n_iterations_(max_n_iterations),
    estimator_id_(estimator_id),
    user_info_size_(user_info_size),
    output_parameters_( output_parameters ),
    output_states_(output_states),
    output_chi_squares_(output_chi_squares),
    output_n_iterations_(output_n_iterations),
    n_parameters_(0),
    data_location_(data_location)
{}

FitInterface::~FitInterface()
{}

void FitInterface::check_sizes()
{
    std::size_t maximum_size = std::numeric_limits< std::size_t >::max();
    
    if (n_fits_ > maximum_size / n_points_ / sizeof(REAL))
    {
        throw std::runtime_error("maximum absolute number of data points exceeded");
    }
    
    if (n_fits_ > maximum_size / n_parameters_ / sizeof(REAL))
    {
        throw std::runtime_error("maximum number of fits and/or parameters exceeded");
    }
}

void FitInterface::configure_info(Info & info, ModelID const model_id)
{
    info.model_id_ = model_id;
    info.n_fits_ = n_fits_;
    info.n_points_ = n_points_;
    info.max_n_iterations_ = max_n_iterations_;
    info.estimator_id_ = estimator_id_;
    info.user_info_size_ = user_info_size_;
    info.n_parameters_ = n_parameters_;
    info.use_constraints_ = constraints_ ? true : false;
    info.use_weights_ = weights_ ? true : false;
    info.data_location_ = data_location_;

    info.set_number_of_parameters_to_fit(parameters_to_fit_);
    info.configure();
}

void FitInterface::fit(ModelID const model_id)
{
    int n_dimensions = 0;
    configure_model(model_id, n_parameters_, n_dimensions);

    check_sizes();

    Info info;
    configure_info(info, model_id);

    LMFit lmfit
    (
        data_,
        weights_,
        info,
        initial_parameters_,
        parameters_to_fit_,
        constraints_,
        constraint_types_,
        user_info_,
        output_parameters_,
        output_states_,
        output_chi_squares_,
        output_n_iterations_
    ) ;
    lmfit.run(tolerance_);
}
