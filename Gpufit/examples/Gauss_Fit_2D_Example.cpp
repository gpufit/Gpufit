#include "../gpufit.h"

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <numeric>
#include <math.h>

void generate_gauss_2d(
    std::vector<float> const & x_coordinates,
    std::vector<float> const & y_coordinates,
    std::vector<float> const & gauss_params, 
    std::vector<float> & output_values)
{
	// Generates a Gaussian 2D function at a set of X and Y coordinates.  The Gaussian is defined by
    // an array of five parameters.
	
	// x_coordinates: Vector of X coordinates.
	// y_coordinates: Vector of Y coordinates.
	// gauss_params:  Vector of function parameters.
	// output_values: Output vector containing the values of the Gaussian function at the
	//                corresponding X, Y coordinates.
	
	// gauss_params[0]: Amplitude
	// gauss_params[1]: Center X position
	// guass_params[2]: Center Y position
	// gauss_params[3]: Gaussian width (standard deviation)
	// gauss_params[4]: Baseline offset
	
	// This code assumes that x_coordinates.size == y_coordinates.size == output_values.size
	
	for (size_t i = 0; i < x_coordinates.size(); i++)
	{
		
		float arg = -(   (x_coordinates[i] - gauss_params[1]) * (x_coordinates[i] - gauss_params[1]) 
		               + (y_coordinates[i] - gauss_params[2]) * (y_coordinates[i] - gauss_params[2])   ) 
					 / (2.f * gauss_params[3] * gauss_params[3]);
					 
		output_values[i] = gauss_params[0] * exp(arg) + gauss_params[4];
		
	}
}

void gauss_fit_2d_example()
{
	/*
        This example generates test data in form of 10000 two dimensional Gaussian
        peaks with the size of 5x5 data points per peak. It is noised by Poisson
        distributed noise. The initial guesses were randomized, within a specified
        range of the true value. The GAUSS_2D model is fitted to the test data sets
        using the MLE estimator.

        The console output shows
         - the execution time,
         - the ratio of converged fits including ratios of not converged fits for 
           different reasons,
         - the values of the true parameters and the mean values of the fitted
           parameters including their standard deviation,
         - the mean chi square value
         - and the mean number of iterations needed.

		True parameters and noise and number of fits is the same as for the Matlab/Python 2D Gaussian examples.
	*/


	// number of fits, fit points and parameters
	size_t const n_fits = 10000;
	size_t const size_x = 50;
	size_t const n_points_per_fit = size_x * size_x;
	size_t const n_model_parameters = 5;

	// true parameters (amplitude, center x position, center y position, width, offset)
	std::vector< float > true_parameters{ 10.f, 14.5f, 14.5f, 3.f, 10.f}; 

	// initialize random number generator
	std::mt19937 rng;
	rng.seed(0);
	std::uniform_real_distribution< float> uniform_dist(0, 1);

	// initial parameters (randomized)
	std::vector< float > initial_parameters(n_fits * n_model_parameters);
	for (size_t i = 0; i < n_fits; i++)
	{
		for (size_t j = 0; j < n_model_parameters; j++)
		{
			if (j == 1 || j == 2)
			{
				initial_parameters[i * n_model_parameters + j]
                    = true_parameters[j] + true_parameters[3] 
                    * (-0.2f + 0.4f * uniform_dist(rng));
			}
			else
			{
				initial_parameters[i * n_model_parameters + j]
                    = true_parameters[j] * (0.8f + 0.4f * uniform_dist(rng));
			}
		}
	}

	// generate x and y values
	std::vector< float > x(n_points_per_fit);
	std::vector< float > y(n_points_per_fit);
	for (size_t i = 0; i < size_x; i++)
	{
		for (size_t j = 0; j < size_x; j++) {
			x[i * size_x + j] = static_cast<float>(j);
			y[i * size_x + j] = static_cast<float>(i);
		}
	}

	// generate test data with Poisson noise
	std::vector< float > temp(n_points_per_fit);
	generate_gauss_2d(x, y, true_parameters, temp);

	std::vector< float > data(n_fits * n_points_per_fit);
	for (size_t i = 0; i < n_fits; i++)
	{
		for (size_t j = 0; j < n_points_per_fit; j++)
		{
			std::poisson_distribution< int > poisson_dist(temp[j]);
			data[i * n_points_per_fit + j] = static_cast<float>(poisson_dist(rng));
		}
	}

	// tolerance
	float const tolerance = 0.001f;

	// maximum number of iterations
	int const max_number_iterations = 20;

	// estimator ID
	int const estimator_id = MLE;

	// model ID
	int const model_id = GAUSS_2D;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

	// output parameters
	std::vector< float > output_parameters(n_fits * n_model_parameters);
	std::vector< int > output_states(n_fits);
	std::vector< float > output_chi_square(n_fits);
	std::vector< int > output_number_iterations(n_fits);

	// call to gpufit (C interface)
	std::chrono::high_resolution_clock::time_point time_0 = std::chrono::high_resolution_clock::now();
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
	std::chrono::high_resolution_clock::time_point time_1 = std::chrono::high_resolution_clock::now();

	// check status
	if (status != ReturnState::OK)
	{
		throw std::runtime_error(gpufit_get_last_error());
	}

	// print execution time
	std::cout
        << "execution time "
        << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count() << " ms\n";

	// get fit states
	std::vector< int > output_states_histogram(5, 0);
	for (std::vector< int >::iterator it = output_states.begin(); it != output_states.end(); ++it)
	{
		output_states_histogram[*it]++;
	}

	std::cout << "ratio converged              " << (float)output_states_histogram[0] / n_fits << "\n";
	std::cout << "ratio max iteration exceeded " << (float)output_states_histogram[1] / n_fits << "\n";
	std::cout << "ratio singular hessian       " << (float)output_states_histogram[2] / n_fits << "\n";
	std::cout << "ratio neg curvature MLE      " << (float)output_states_histogram[3] / n_fits << "\n";
	std::cout << "ratio gpu not read           " << (float)output_states_histogram[4] / n_fits << "\n";

	// compute mean of fitted parameters for converged fits
	std::vector< float > output_parameters_mean(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			for (size_t j = 0; j < n_model_parameters; j++)
			{
				output_parameters_mean[j] += output_parameters[i * n_model_parameters + j];
			}
		}
	}
	// normalize
	for (size_t j = 0; j < n_model_parameters; j++)
	{
		output_parameters_mean[j] /= output_states_histogram[0];
	}
	
	// compute std of fitted parameters for converged fits
	std::vector< float > output_parameters_std(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			for (size_t j = 0; j < n_model_parameters; j++)
			{
				output_parameters_std[j]
                    += (output_parameters[i * n_model_parameters + j] - output_parameters_mean[j])
                    *  (output_parameters[i * n_model_parameters + j] - output_parameters_mean[j]);
			}
		}
	}
	// normalize and take square root
	for (size_t j = 0; j < n_model_parameters; j++)
	{
		output_parameters_std[j] = sqrt(output_parameters_std[j] / output_states_histogram[0]);
	}

	// print true value, fitted mean and std for every parameter
	for (size_t j = 0; j < n_model_parameters; j++)
	{
		std::cout
            << "parameter "     << j
            << " true "         << true_parameters[j]
            << " fitted mean "  << output_parameters_mean[j]
            << " std "          << output_parameters_std[j] << "\n";
	}

	// compute mean chi-square for those converged
	float  output_chi_square_mean = 0;
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			output_chi_square_mean += output_chi_square[i];
		}
	}
	output_chi_square_mean /= static_cast<float>(output_states_histogram[0]);
	std::cout << "mean chi square " << output_chi_square_mean << "\n";

	// compute mean number of iterations for those converged
	float  output_number_iterations_mean = 0;
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
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
	gauss_fit_2d_example();

    std::cout << std::endl << "Example completed!" << std::endl;
    std::cout << "Press ENTER to exit" << std::endl;
    std::getchar();

	return 0;
}
