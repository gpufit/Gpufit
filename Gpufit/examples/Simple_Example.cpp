#include "../gpufit.h"
#include <iostream>
#include <vector>

void simple_example()
{
	/*
		Simple example demonstrating a minimal call of all needed parameters to
        the C interface. It can be built and executed, but in this exeample
        gpufit doesn't do anything useful and it doesn't yield meaningful
        output. No test data is generated. The values of the input data vector
        and the initial fit parameters vector are set to 0.

        This example can be devided in three parts:
            - definition of input and output parameters
            - call to gpufit
            - status check
	*/

    /*************** definition of input and output parameters  ***************/

	// number of fits, number of points per fit
	size_t const number_fits = 10;
	size_t const number_points = 10;

	// model ID and number of parameter
	int const model_id = GAUSS_1D;
	size_t const number_parameters = 4;

	// initial parameters
	std::vector< float > initial_parameters(number_fits * number_parameters);

	// data
	std::vector< float > data(number_points * number_fits);

	// tolerance
	float const tolerance = 0.001f;

	// maximal number of iterations
	int const max_number_iterations = 10;

	// estimator ID
	int const estimator_id = LSE;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(number_parameters, 1);

	// output parameters
	std::vector< float > output_parameters(number_fits * number_parameters);
	std::vector< int > output_states(number_fits);
	std::vector< float > output_chi_square(number_fits);
	std::vector< int > output_number_iterations(number_fits);

    /***************************** call to gpufit  ****************************/

	int const status = gpufit
        (
            number_fits,
            number_points,
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
            output_number_iterations.data(),
            0
        );

    /****************************** status check  *****************************/

	if (status != STATUS_OK)
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
