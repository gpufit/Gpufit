#include "../gpufit.h"
#include <iostream>
#include <vector>

void simple_example()
{
	/*
		This example demonstrates a simple, minimal program containing all 
		of the required parameters for a call to the Gpufit function.  The example 
		can be built and executed within the project environment. Please note that 
		this code does not actually do anything other than make a single call to 
		gpufit().

		In the first section of the code, the *model ID* is set, memory space for 
		initial parameters and data values is allocated, the *fit tolerance* is set, 
		the *maximum number of iterations* is set, the *estimator ID* is set, and 
		the *parameters to fit array* is initialized.  Note that in most applications, 
		the data array will already exist and it will be unnecessary to allocate 
		additional space for data.  In this example, the *parameters to fit* array 
		is initialized to all ones, indicating that all model parameters should be 
		adjusted in the fit.
	*/

    /*************** definition of input and output parameters  ***************/

	// number of fits, number of points per fit
	std::size_t const n_fits = 10;
	std::size_t const n_points_per_fit = 10;

	// model ID and number of model parameters
	int const model_id = GAUSS_1D;
	std::size_t const n_model_parameters = 4;

	// initial parameters
	std::vector< REAL > initial_parameters(n_fits * n_model_parameters);

	// data
	std::vector< REAL > data(n_points_per_fit * n_fits);

	// tolerance
	REAL const tolerance = 0.001f;

	// maximum number of iterations
	int const max_number_iterations = 10;

	// estimator ID
	int const estimator_id = LSE;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

	// output parameters
	std::vector< REAL > output_parameters(n_fits * n_model_parameters);
	std::vector< int > output_states(n_fits);
	std::vector< REAL > output_chi_square(n_fits);
	std::vector< int > output_number_iterations(n_fits);

    /***************************** call to gpufit  ****************************/

	int const status = gpufit
        (
            n_fits,
            n_points_per_fit,
            data.data(),
            0,
            model_id,
            initial_parameters.data(),
            tolerance,
            max_number_iterations,
            parameters_to_fit.data(),
            estimator_id,
            0,
            0,
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_number_iterations.data()
        );

    /****************************** status check  *****************************/

	if (status != ReturnState::OK)
	{
		throw std::runtime_error(gpufit_get_last_error());
	}
}


int main(int argc, char *argv[])
{
	simple_example();

    std::cout << std::endl << "Example completed!" << std::endl;
    std::cout << "Press ENTER to exit" << std::endl;
    std::getchar();
	
	return 0;
}
