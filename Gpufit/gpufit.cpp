#include "gpufit.h"
#include "interface.h"

#include <string>

std::string last_error ;

int gpufit
(
    size_t n_fits,
    size_t n_points,
    float * data,
    float * weights,
    int model_id,
    float * initial_parameters,
    float tolerance,
    int max_n_iterations,
    int * parameters_to_fit,
    int estimator_id,
    size_t user_info_size,
    char * user_info,
    float * output_parameters,
    int * output_states,
    float * output_chi_squares,
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
        *((size_t *) argv[0]),
        *((size_t *) argv[1]),
        (float *) argv[2],
        (float *) argv[3],
        *((int *) argv[4]),
        (float *) argv[5],
        *((float *) argv[6]),
        *((int *) argv[7]),
        (int *) argv[8],
        *((int *) argv[9]),
        *((size_t *) argv[10]),
        (char *) argv[11],
        (float *) argv[12],
        (int *) argv[13],
        (float *) argv[14],
        (int *) argv[15]);

}