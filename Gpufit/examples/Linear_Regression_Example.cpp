#include "../gpufit.h"

#include <vector>
#include <random>
#include <iostream>
#include <math.h>

void linear_regression_example()
{
    /*
    This example generates test data in form of 10000 one dimensional linear
    curves with the size of 20 data points per curve. It is noised by normal
    distributed noise. The initial guesses were randomized, within a specified
    range of the true value. The LINEAR_1D model is fitted to the test data sets
    using the LSE estimator. The optional parameter user_info is used to pass 
    custom x positions of the data sets. The same x position values are used for
    every fit.

    The console output shows
    - the ratio of converged fits including ratios of not converged fits for
      different reasons,
    - the values of the true parameters and the mean values of the fitted
      parameters including their standard deviation,
    - the mean chi square value
    - and the mean number of iterations needed.
    */

	// number of fits, fit points and parameters
	size_t const number_fits = 10000;
	size_t const number_points = 20;
	size_t const number_parameters = 2;

	// custom x positions for the data points of every fit, stored in user info
	std::vector< float > user_info(number_points);
	for (size_t i = 0; i < number_points; i++)
	{
		user_info[i] = static_cast<float>(pow(2, i));
	}

	// size of user info in bytes
	size_t const user_info_size = number_points * sizeof(float); 

	// initialize random number generator
	std::mt19937 rng;
	rng.seed(0);
	std::uniform_real_distribution< float > uniform_dist(0, 1);
	std::normal_distribution< float > normal_dist(0, 1);

	// true parameters
	std::vector< float > true_parameters { 5, 2 }; // offset, slope

	// initial parameters (randomized)
	std::vector< float > initial_parameters(number_fits * number_parameters);
	for (size_t i = 0; i != number_fits; i++)
	{
		// random offset
		initial_parameters[i * number_parameters + 0] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
		// random slope
		initial_parameters[i * number_parameters + 1] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
	}

	// generate data
	std::vector< float > data(number_points * number_fits);
	for (size_t i = 0; i != data.size(); i++)
	{
		size_t j = i / number_points; // the fit
		size_t k = i % number_points; // the position within a fit

		float x = user_info[k];
		float y = true_parameters[0] + x * true_parameters[1];
		data[i] = y + normal_dist(rng);
	}

	// tolerance
	float const tolerance = 0.001f;

	// maximal number of iterations
	int const max_number_iterations = 20;

	// estimator ID
	int const estimator_id = LSE;

	// model ID
	int const model_id = LINEAR_1D;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(number_parameters, 1);

	// output parameters
	std::vector< float > output_parameters(number_fits * number_parameters);
	std::vector< int > output_states(number_fits);
	std::vector< float > output_chi_square(number_fits);
	std::vector< int > output_number_iterations(number_fits);

	// call to gpufit (C interface)
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
            user_info_size,
            reinterpret_cast< char * >( user_info.data() ),
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_number_iterations.data(),
            0
        );

	// check status
	if (status != STATUS_OK)
	{
		throw std::runtime_error(gpufit_get_last_error());
	}

	// get fit states
	std::vector< int > output_states_histogram(5, 0);
	for (std::vector< int >::iterator it = output_states.begin(); it != output_states.end(); ++it)
	{
		output_states_histogram[*it]++;
	}

	std::cout << "ratio converged              " << (float) output_states_histogram[0] / number_fits << "\n";
	std::cout << "ratio max iteration exceeded " << (float) output_states_histogram[1] / number_fits << "\n";
	std::cout << "ratio singular hessian       " << (float) output_states_histogram[2] / number_fits << "\n";
	std::cout << "ratio neg curvature MLE      " << (float) output_states_histogram[3] / number_fits << "\n";
	std::cout << "ratio gpu not read           " << (float) output_states_histogram[4] / number_fits << "\n";

	// compute mean fitted parameters for converged fits
	std::vector< float > output_parameters_mean(number_parameters, 0);
	for (size_t i = 0; i != number_fits; i++)
	{
		if (output_states[i] == STATE_CONVERGED)
		{
			// add offset
			output_parameters_mean[0] += output_parameters[i * number_parameters + 0];
			// add slope
			output_parameters_mean[1] += output_parameters[i * number_parameters + 1];
		}
	}
	output_parameters_mean[0] /= output_states_histogram[0];
	output_parameters_mean[1] /= output_states_histogram[0];

	// compute std of fitted parameters for converged fits
	std::vector< float > output_parameters_std(number_parameters, 0);
	for (size_t i = 0; i != number_fits; i++)
	{
		if (output_states[i] == STATE_CONVERGED)
		{
			// add squared deviation for offset
			output_parameters_std[0] += (output_parameters[i * number_parameters + 0] - output_parameters_mean[0]) * (output_parameters[i * number_parameters + 0] - output_parameters_mean[0]);
			// add squared deviation for slope
			output_parameters_std[1] += (output_parameters[i * number_parameters + 1] - output_parameters_mean[1]) * (output_parameters[i * number_parameters + 1] - output_parameters_mean[1]);
		}
	}
	// divide and take square root
	output_parameters_std[0] = sqrt(output_parameters_std[0] / output_states_histogram[0]);
	output_parameters_std[1] = sqrt(output_parameters_std[1] / output_states_histogram[0]);

	// print mean and std
	std::cout << "offset  true " << true_parameters[0] << " mean " << output_parameters_mean[0] << " std " << output_parameters_std[0] << "\n";
	std::cout << "slope   true " << true_parameters[1] << " mean " << output_parameters_mean[1] << " std " << output_parameters_std[1] << "\n";

	// compute mean chi-square for those converged
	float  output_chi_square_mean = 0;
	for (size_t i = 0; i != number_fits; i++)
	{
		if (output_states[i] == STATE_CONVERGED)
		{
			output_chi_square_mean += output_chi_square[i];
		}
	}
	output_chi_square_mean /= static_cast<float>(output_states_histogram[0]);
	std::cout << "mean chi square " << output_chi_square_mean << "\n";

	// compute mean number of iterations for those converged
	float  output_number_iterations_mean = 0;
	for (size_t i = 0; i != number_fits; i++)
	{
		if (output_states[i] == STATE_CONVERGED)
		{
			output_number_iterations_mean += static_cast<float>(output_number_iterations[i]);
		}
	}

	// normalize
	output_number_iterations_mean /= static_cast<float>(output_states_histogram[0]);
	std::cout << "mean number of iterations " << output_number_iterations_mean << "\n";
}


int main(int argc, char *argv[])
{
	linear_regression_example();

    std::cout << std::endl << "Example completed!" << std::endl;
    std::cout << "Press ENTER to exit" << std::endl;
    std::getchar();
	
	return 0;
}
