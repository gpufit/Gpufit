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
	std::size_t const n_fits = 10000;
	std::size_t const n_points_per_fit = 20;
	std::size_t const n_model_parameters = 2;

	// custom x positions for the data points of every fit, stored in user info
	std::vector< REAL > user_info(n_points_per_fit);
	for (std::size_t i = 0; i < n_points_per_fit; i++)
	{
		user_info[i] = static_cast<REAL>(pow(2, i));
	}

	// size of user info in bytes
	std::size_t const user_info_size = n_points_per_fit * sizeof(REAL); 

	// initialize random number generator
	std::mt19937 rng;
	rng.seed(0);
	std::uniform_real_distribution< REAL > uniform_dist(0, 1);
	std::normal_distribution< REAL > normal_dist(0, 1);

	// true parameters
	std::vector< REAL > true_parameters { 5, 2 }; // offset, slope

	// initial parameters (randomized)
	std::vector< REAL > initial_parameters(n_fits * n_model_parameters);
	for (std::size_t i = 0; i != n_fits; i++)
	{
		// random offset
		initial_parameters[i * n_model_parameters + 0] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
		// random slope
		initial_parameters[i * n_model_parameters + 1] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
	}

	// generate data
	std::vector< REAL > data(n_points_per_fit * n_fits);
	for (std::size_t i = 0; i != data.size(); i++)
	{
		std::size_t j = i / n_points_per_fit; // the fit
		std::size_t k = i % n_points_per_fit; // the position within a fit

		REAL x = user_info[k];
		REAL y = true_parameters[0] + x * true_parameters[1];
		data[i] = y + normal_dist(rng);
	}

	// tolerance
	REAL const tolerance = 0.001f;

	// maximum number of iterations
	int const max_number_iterations = 20;

	// estimator ID
	int const estimator_id = LSE;

	// model ID
	int const model_id = LINEAR_1D;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

	// output parameters
	std::vector< REAL > output_parameters(n_fits * n_model_parameters);
	std::vector< int > output_states(n_fits);
	std::vector< REAL > output_chi_square(n_fits);
	std::vector< int > output_number_iterations(n_fits);

	// call to gpufit (C interface)
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
            user_info_size,
            reinterpret_cast< char * >( user_info.data() ),
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_number_iterations.data()
        );

	// check status
	if (status != ReturnState::OK)
	{
		throw std::runtime_error(gpufit_get_last_error());
	}

	// get fit states
	std::vector< int > output_states_histogram(5, 0);
	for (std::vector< int >::iterator it = output_states.begin(); it != output_states.end(); ++it)
	{
		output_states_histogram[*it]++;
	}

	std::cout << "ratio converged              " << (REAL) output_states_histogram[0] / n_fits << "\n";
	std::cout << "ratio max iteration exceeded " << (REAL) output_states_histogram[1] / n_fits << "\n";
	std::cout << "ratio singular hessian       " << (REAL) output_states_histogram[2] / n_fits << "\n";
	std::cout << "ratio neg curvature MLE      " << (REAL) output_states_histogram[3] / n_fits << "\n";
	std::cout << "ratio gpu not read           " << (REAL) output_states_histogram[4] / n_fits << "\n";

	// compute mean fitted parameters for converged fits
	std::vector< REAL > output_parameters_mean(n_model_parameters, 0);
	for (std::size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			// add offset
			output_parameters_mean[0] += output_parameters[i * n_model_parameters + 0];
			// add slope
			output_parameters_mean[1] += output_parameters[i * n_model_parameters + 1];
		}
	}
	output_parameters_mean[0] /= output_states_histogram[0];
	output_parameters_mean[1] /= output_states_histogram[0];

	// compute std of fitted parameters for converged fits
	std::vector< REAL > output_parameters_std(n_model_parameters, 0);
	for (std::size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			// add squared deviation for offset
			output_parameters_std[0] += (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]) * (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]);
			// add squared deviation for slope
			output_parameters_std[1] += (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]) * (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]);
		}
	}
	// divide and take square root
	output_parameters_std[0] = sqrt(output_parameters_std[0] / output_states_histogram[0]);
	output_parameters_std[1] = sqrt(output_parameters_std[1] / output_states_histogram[0]);

	// print mean and std
	std::cout << "offset  true " << true_parameters[0] << " mean " << output_parameters_mean[0] << " std " << output_parameters_std[0] << "\n";
	std::cout << "slope   true " << true_parameters[1] << " mean " << output_parameters_mean[1] << " std " << output_parameters_std[1] << "\n";

	// compute mean chi-square for those converged
	REAL  output_chi_square_mean = 0;
	for (std::size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			output_chi_square_mean += output_chi_square[i];
		}
	}
	output_chi_square_mean /= static_cast<REAL>(output_states_histogram[0]);
	std::cout << "mean chi square " << output_chi_square_mean << "\n";

	// compute mean number of iterations for those converged
	REAL  output_number_iterations_mean = 0;
	for (std::size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			output_number_iterations_mean += static_cast<REAL>(output_number_iterations[i]);
		}
	}

	// normalize
	output_number_iterations_mean /= static_cast<REAL>(output_states_histogram[0]);
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
