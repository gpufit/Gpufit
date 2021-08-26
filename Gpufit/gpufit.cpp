#include "gpufit.h"
#include "interface.h"

#include <string>

std::string last_error ;

int gpufit
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
        NULL,
        NULL,
        user_info,
        user_info_size,
        output_parameters,
        output_states,
        output_chi_squares,
        output_n_iterations,
        HOST);

    fi.fit(static_cast<ModelID>(model_id));

    return ReturnState::OK ;
}
catch( std::exception & exception )
{
    last_error = exception.what() ;

    return ReturnState::ERROR ;
}
catch( ... )
{
    last_error = "unknown error" ;

    return ReturnState::ERROR;
}

int gpufit_constrained
(
    std::size_t n_fits,
    std::size_t n_points,
    REAL * data,
    REAL * weights,
    int model_id,
    REAL * initial_parameters,
    REAL * constraints,
    int * constraint_types,
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
        constraints,
        constraint_types,
        user_info,
        user_info_size,
        output_parameters,
        output_states,
        output_chi_squares,
        output_n_iterations,
        HOST);

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
    last_error = "unknown error";

    return ReturnState::ERROR;
}

int gpufit_cuda_interface
(
    std::size_t n_fits,
    std::size_t n_points,
    REAL * gpu_data,
    REAL * gpu_weights,
    int model_id,
    REAL tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    std::size_t user_info_size,
    char * gpu_user_info,
    REAL * gpu_fit_parameters,
    int * gpu_output_states,
    REAL * gpu_output_chi_squares,
    int * gpu_output_n_iterations
)
try
{
    FitInterface fi(
        gpu_data,
        gpu_weights,
        n_fits,
        static_cast<int>(n_points),
        tolerance,
        max_n_iterations,
        static_cast<EstimatorID>(estimator_id),
        gpu_fit_parameters,
        parameters_to_fit,
        NULL,
        NULL,
        gpu_user_info,
        user_info_size,
        gpu_fit_parameters,
        gpu_output_states,
        gpu_output_chi_squares,
        gpu_output_n_iterations,
        DEVICE);

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
    last_error = "unknown error";

    return ReturnState::ERROR;
}

int gpufit_constrained_cuda_interface
(
    std::size_t n_fits,
    std::size_t n_points,
    REAL* gpu_data,
    REAL* gpu_weights,
    int model_id,
    REAL tolerance,
    int max_n_iterations,
    int* parameters_to_fit,
    REAL* gpu_constraints,
    int* constraint_types,
    int estimator_id,
    std::size_t user_info_size,
    char* gpu_user_info,
    REAL* gpu_fit_parameters,
    int* gpu_output_states,
    REAL* gpu_output_chi_squares,
    int* gpu_output_n_iterations
)
try
{
    FitInterface fi(
        gpu_data,
        gpu_weights,
        n_fits,
        static_cast<int>(n_points),
        tolerance,
        max_n_iterations,
        static_cast<EstimatorID>(estimator_id),
        gpu_fit_parameters,
        parameters_to_fit,
        gpu_constraints,
        constraint_types,
        gpu_user_info,
        user_info_size,
        gpu_fit_parameters,
        gpu_output_states,
        gpu_output_chi_squares,
        gpu_output_n_iterations,
        DEVICE);

    fi.fit(static_cast<ModelID>(model_id));

    return ReturnState::OK;
}
catch (std::exception& exception)
{
    last_error = exception.what();

    return ReturnState::ERROR;
}
catch (...)
{
    last_error = "unknown error";

    return ReturnState::ERROR;
}

char const * gpufit_get_last_error()
{
    return last_error.c_str() ;
}

int gpufit_cuda_available()
{
	// Returns 1 if CUDA is available and 0 otherwise
	try
	{
		getDeviceCount();
		return 1;
	}
	catch (std::exception & exception)
	{
		last_error = exception.what();

		return 0;
	}
}

int gpufit_get_cuda_version(int * runtime_version, int * driver_version)
{
    try
    {
        cudaRuntimeGetVersion(runtime_version);
        cudaDriverGetVersion(driver_version);
        return ReturnState::OK;
    }
    catch (std::exception & exception)
    {
        last_error = exception.what();

        return ReturnState::ERROR;
    }
}

int gpufit_portable_interface(int argc, void *argv[])
{

    return gpufit(
        *((std::size_t *) argv[0]),
        *((std::size_t *) argv[1]),
        (REAL *) argv[2],
        (REAL *) argv[3],
        *((int *) argv[4]),
        (REAL *) argv[5],
        *((REAL *) argv[6]),
        *((int *) argv[7]),
        (int *) argv[8],
        *((int *) argv[9]),
        *((std::size_t *) argv[10]),
        (char *) argv[11],
        (REAL *) argv[12],
        (int *) argv[13],
        (REAL *) argv[14],
        (int *) argv[15]);

}

int gpufit_constrained_portable_interface(int argc, void *argv[])
{

    return gpufit_constrained(
        *((std::size_t *) argv[0]),
        *((std::size_t *) argv[1]),
        (REAL *) argv[2],
        (REAL *) argv[3],
        *((int *) argv[4]),
        (REAL *) argv[5],
        (REAL *) argv[6],
        (int *) argv[7],
        *((REAL *) argv[8]),
        *((int *) argv[9]),
        (int *) argv[10],
        *((int *) argv[11]),
        *((std::size_t *) argv[12]),
        (char *) argv[13],
        (REAL *) argv[14],
        (int *) argv[15],
        (REAL *) argv[16],
        (int *) argv[17]);

}