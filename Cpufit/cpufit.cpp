#include "cpufit.h"
#include "../Gpufit/constants.h"
#include "interface.h"

#include <string>

std::string last_error ;

int cpufit
(
    std::size_t n_fits,
    std::size_t n_points,
    REAL * data,
    REAL * weights,
    int model_id,
    REAL * initial_parameters,
    REAL tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    std::size_t user_info_size,
    char * user_info,
    REAL * output_parameters,
    int * output_states,
    REAL * output_chi_squares,
    int * output_n_iterations
)
try
{
    FitInterface fi(
        data,
        weights,
        n_fits,
        static_cast<int>(n_points),
        tolerance,
        max_n_iterations,
        static_cast<EstimatorID>(estimator_id),
        initial_parameters,
        parameters_to_fit,
        user_info,
        user_info_size,
        output_parameters,
        output_states,
        output_chi_squares,
        output_n_iterations);

    fi.fit(static_cast<ModelID>(model_id));

    return ReturnState::OK;
}
catch (std::exception & exception)
{
    last_error = exception.what();

    return ReturnState::ERROR;
}
catch (...)
{
    last_error = "Unknown Error";

    return ReturnState::ERROR;
}

char const * cpufit_get_last_error()
{
    return last_error.c_str();
}
