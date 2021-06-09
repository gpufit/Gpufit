#include <limits>
#include <stdexcept>

#include "cpufit.h"
#include "interface.h"

FitInterface::FitInterface(
    REAL const * data,
    REAL const * weights,
    std::size_t n_fits,
    int n_points,
    REAL tolerance,
    int max_n_iterations,
    EstimatorID estimator_id,
    REAL const * initial_parameters,
    int const * parameters_to_fit,
    char * user_info,
    std::size_t user_info_size,
    REAL * output_parameters,
    int * output_states,
    REAL * output_chi_squares,
    int * output_n_iterations) :
    data_(data),
    weight_(weights),
    n_fits_(n_fits),
    n_points_(n_points),
    tolerance_(tolerance),
    max_n_iterations_(max_n_iterations),
    estimator_id_(estimator_id),
    initial_parameters_(initial_parameters),
    parameters_to_fit_(parameters_to_fit),
    user_info_(user_info),
    user_info_size_(user_info_size),
    output_parameters_(output_parameters),
    output_states_(output_states),
    output_chi_squares_(output_chi_squares),
    output_n_iterations_(output_n_iterations),
    n_parameters_(0)
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

    info.set_number_of_parameters_to_fit(parameters_to_fit_);
}

void FitInterface::set_number_of_parameters(ModelID const model_id)
{
    switch (model_id)
    {
    case GAUSS_1D:
        n_parameters_ = 4;
        break;
    case GAUSS_2D:
        n_parameters_ = 5;
        break;
    case GAUSS_2D_ELLIPTIC:
        n_parameters_ = 6;
        break;
    case GAUSS_2D_ROTATED:
        n_parameters_ = 7;
        break;
    case CAUCHY_2D_ELLIPTIC:
        n_parameters_ = 6;
        break;
    case LINEAR_1D:
        n_parameters_ = 2;
        break;
    case FLETCHER_POWELL_HELIX:
        n_parameters_ = 3;
        break;
    case BROWN_DENNIS:
        n_parameters_ = 4;
        break;
    case SPLINE_1D:
        n_parameters_ = 3;
        break;
    case SPLINE_2D:
        n_parameters_ = 4;
        break;
    case SPLINE_3D:
        n_parameters_ = 5;
        break;
    case SPLINE_3D_MULTICHANNEL:
        n_parameters_ = 5;
        break;
    case SPLINE_3D_PHASE_MULTICHANNEL:
        n_parameters_ = 6;
        break;
    default:
        throw std::runtime_error("unknown model ID");
    }
}

void FitInterface::fit(ModelID const model_id)
{
    set_number_of_parameters(model_id);

    check_sizes();

    Info info;
    configure_info(info, model_id);

    LMFit lmfit(
        data_,
        weight_,
        info,
        initial_parameters_,
        parameters_to_fit_,
        user_info_,
        output_parameters_,
        output_states_,
        output_chi_squares_,
        output_n_iterations_);

    lmfit.run(tolerance_);
}
