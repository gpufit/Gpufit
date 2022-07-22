#include "../../Gpufit/gpufit.h"

#include <vector>
#include <random>
#include <iostream>
#include <math.h>
#include <complex>
using namespace std;

void liver_fat_three()
{
	// variables
	float const exp = 2.71828;
	float const pi =  3.14159;
	std::complex<REAL> expC = 2.71828;
	std::complex<REAL> piC = 3.14159;

	// estimator ID
	int const estimator_id = LSE;

	// model ID
	int const model_id = LIVER_FAT_THREE;

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

	// Signal to Noise Ratio
	float snr = 5000;
	// number of fits, fit points and parameters
	size_t const n_fits = 100; //10000
	size_t const n_points_per_fit = 6;

	size_t const n_model_parameters = 3;

// just in case it was possible to upload manually the echotimes
	/*
	std::vector<REAL> TEn(6);
	float echo_time;
	std::cout << "Please enter in echo times: ";
	for (size_t i = 0; i < 6 ; i++)
	{
		std::cin >> echo_time;
		TEn.push_back (echo_time);
	}
	*/

	// custom x positions for the data points of every fit, stored in user info
	REAL TEn[] = {1.23, 2.48, 3.65, 4.84, 6.03, 7.22};
	std::complex<REAL> TEnC[] = {1.23, 2.48, 3.65, 4.84, 6.03, 7.22};
	std::vector< REAL > user_info(n_points_per_fit);
	// std::cout << "Enter in echo times: "
	for (size_t i = 0; i < n_points_per_fit; i++)
	{
		user_info[i] = static_cast<REAL>(TEn[i]);
	}

	// size of user info in bytes
	size_t const user_info_size = n_points_per_fit * sizeof(REAL);

	// true parameters
	std::vector< REAL > true_parameters { 210, 175, .5};

	float sigma =  (true_parameters[0] + true_parameters[1]) / snr;

	// initialize random number generator
	std::mt19937 rng;
	rng.seed(0);
	std::uniform_real_distribution< REAL > uniform_dist(0, 1);
	std::normal_distribution< REAL > normal_dist(0, sigma);

	// initial parameters (randomized)
	std::vector< REAL > initial_parameters(n_fits * n_model_parameters);
	for (size_t i = 0; i != n_fits; i++)
	{
		// random 1st parameter
		initial_parameters[i * n_model_parameters + 0] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
		// random 2nd parameter
		initial_parameters[i * n_model_parameters + 1] = true_parameters[1] * (0.8f + 0.4f * uniform_dist(rng));
		// random 3rd parameter
		initial_parameters[i * n_model_parameters + 2] = true_parameters[2] * (0.8f + 0.4f * uniform_dist(rng));

		std::cout << "parameter 0 + noise            " << (REAL) initial_parameters[i * n_model_parameters + 0] << "\n";
		std::cout << "parameter 1 + noise            " << (REAL) initial_parameters[i * n_model_parameters + 1] << "\n";
		std::cout << "parameter 2 + noise            " << (REAL) initial_parameters[i * n_model_parameters + 2] << "\n";
	}

	// Complex Number builder
	// First calculate C_n
	// maybe define C_n as a real and imaginary part
	std::complex<REAL> const ppm_list[] = {-0.4764702, -0.4253742, -0.3883296, -0.332124, -0.3040212, -0.2375964, 0.0868632};
	std::complex<REAL> const weight_list[] = {0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04};
	std::complex<REAL> C_n = std::complex<REAL>(0.f, 0.f);
	std::complex<REAL> j = std::complex<REAL> (0.f, 1);



	// generate data
	std::vector< REAL > data(n_points_per_fit * n_fits);
	// Loop over all TEs
	for (size_t i = 0; i != data.size(); i++)
	{
		//size_t j = i / n_points_per_fit; // the fit
		size_t k = i % n_points_per_fit; // the position within a fit

		// C_n calculation
		C_n = std::complex<REAL>(0.f, 0.f);
		//Loop over all ppm/weight factors
		for (int h =0; h < 7; h++)
		{
			// weight_list * e ^ (j * 2 * pi * ppm_list * TEn)
			C_n += weight_list[h] * pow(expC, (j * 2.0f * piC * ppm_list[h] * TEnC[k]));
		}

		REAL x = user_info[k];
		std::cout << "Complex Number: " << C_n << "\n";
		REAL y = abs((true_parameters[0] + C_n * true_parameters[1]) * pow(exp, (-1 * true_parameters[2] * x)));
		float rician_noise = sqrt(pow(normal_dist(rng),2) + pow(normal_dist(rng),2));
		data[i] = y + rician_noise;
		std::cout << "y             " << (REAL) y << "\n";
		std::cout << "rician noise  " << (REAL) rician_noise << "\n";
		std::cout << "y with noise  " << (REAL) data[i] << "\n";
	}

	// tolerance
	REAL const tolerance = 10e-15f;

	// maximum number of iterations
	int const max_number_iterations = 200;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

//error occurs if in models.cuh the parameters are wrong
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
// not sure how much from this area needs to be fixed
	// compute mean fitted parameters for converged fits
	std::vector< REAL > output_parameters_mean(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
// I"m pretty sure this needs to be edited for exponential functions
		if (output_states[i] == FitState::CONVERGED)
		{
			// add 1st parameter
			output_parameters_mean[0] += output_parameters[i * n_model_parameters + 0];
			// add 2nd parameter
			output_parameters_mean[1] += output_parameters[i * n_model_parameters + 1];
			// add 3rd parameter
			output_parameters_mean[2] += output_parameters[i * n_model_parameters + 2];
		}
	}
	output_parameters_mean[0] /= output_states_histogram[0];
	output_parameters_mean[1] /= output_states_histogram[0];
	output_parameters_mean[2] /= output_states_histogram[0];

	// compute std of fitted parameters for converged fits
	std::vector< REAL > output_parameters_std(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			// add squared deviation for 1st parameter
			output_parameters_std[0] += (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]) * (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]);
			// add squared deviation for 2nd parameter
			output_parameters_std[1] += (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]) * (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]);
			// add squared deviation for 3rd parameter
			output_parameters_std[2] += (output_parameters[i * n_model_parameters + 2] - output_parameters_mean[2]) * (output_parameters[i * n_model_parameters + 2] - output_parameters_mean[2]);
		}
	}
	// divide and take square root
	output_parameters_std[0] = sqrt(output_parameters_std[0] / output_states_histogram[0]);
	output_parameters_std[1] = sqrt(output_parameters_std[1] / output_states_histogram[0]);
	output_parameters_std[2] = sqrt(output_parameters_std[2] / output_states_histogram[0]);

	// print mean and std
	std::cout << "parameter 0   true " << true_parameters[0] << " mean " << output_parameters_mean[0] << " std " << output_parameters_std[0] << "\n";
	std::cout << "parameter 1   true " << true_parameters[1] << " mean " << output_parameters_mean[1] << " std " << output_parameters_std[1] << "\n";
	std::cout << "parameter 2   true " << true_parameters[2] << " mean " << output_parameters_mean[2] << " std " << output_parameters_std[2] << "\n";

	// compute mean chi-square for those converged
	REAL  output_chi_square_mean = 0;
	for (size_t i = 0; i != n_fits; i++)
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
	for (size_t i = 0; i != n_fits; i++)
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

void liver_fat_four()
{
	// variables
	float const exp = 2.71828;
	float const pi =  3.14159;
	std::complex<REAL> expC = 2.71828;
	std::complex<REAL> piC = 3.14159;

	// estimator ID
	int const estimator_id = LSE;

	// model ID
	int const model_id = LIVER_FAT_FOUR;

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

	// Signal to Noise Ratio
	float snr = 5000;
	// number of fits, fit points and parameters
	size_t const n_fits = 100; //10000
	size_t const n_points_per_fit = 6;

	size_t const n_model_parameters = 4;

// just in case it was possible to upload manually the echotimes
	/*
	std::vector<REAL> TEn(6);
	float echo_time;
	std::cout << "Please enter in echo times: ";
	for (size_t i = 0; i < 6 ; i++)
	{
		std::cin >> echo_time;
		TEn.push_back (echo_time);
	}
	*/

	// custom x positions for the data points of every fit, stored in user info
	REAL TEn[] = {1.23, 2.48, 3.65, 4.84, 6.03, 7.22};
	std::complex<REAL> TEnC[] = {1.23, 2.48, 3.65, 4.84, 6.03, 7.22};
	std::vector< REAL > user_info(n_points_per_fit);
	// std::cout << "Enter in echo times: "
	for (size_t i = 0; i < n_points_per_fit; i++)
	{
		user_info[i] = static_cast<REAL>(TEn[i]);
	}

	// size of user info in bytes
	size_t const user_info_size = n_points_per_fit * sizeof(REAL);

	// true parameters
	std::vector< REAL > true_parameters { 210, 30, .5, .3};

	float sigma =  (true_parameters[0] + true_parameters[1]) / snr;

	// initialize random number generator
	std::mt19937 rng;
	rng.seed(0);
	std::uniform_real_distribution< REAL > uniform_dist(0, 1);
	std::normal_distribution< REAL > normal_dist(0, sigma);

	// initial parameters (randomized)
	std::vector< REAL > initial_parameters(n_fits * n_model_parameters);
	for (size_t i = 0; i != n_fits; i++)
	{
		// random 1st parameter
		initial_parameters[i * n_model_parameters + 0] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
		// random 2nd parameter
		initial_parameters[i * n_model_parameters + 1] = true_parameters[1] * (0.8f + 0.4f * uniform_dist(rng));
		// random 3rd parameter
		initial_parameters[i * n_model_parameters + 2] = true_parameters[2] * (0.8f + 0.4f * uniform_dist(rng));
		// random 4th parameter
		initial_parameters[i * n_model_parameters + 3] = true_parameters[3] * (0.8f + 0.4f * uniform_dist(rng));

		std::cout << "parameter 0 + noise            " << (REAL) initial_parameters[i * n_model_parameters + 0] << "\n";
		std::cout << "parameter 1 + noise            " << (REAL) initial_parameters[i * n_model_parameters + 1] << "\n";
		std::cout << "parameter 2 + noise            " << (REAL) initial_parameters[i * n_model_parameters + 2] << "\n";
		std::cout << "parameter 3 + noise            " << (REAL) initial_parameters[i * n_model_parameters + 3] << "\n";
	}

	// Complex Number builder
	// First calculate C_n
	// maybe define C_n as a real and imaginary part
	std::complex<REAL> const ppm_list[] = {-0.4764702, -0.4253742, -0.3883296, -0.332124, -0.3040212, -0.2375964, 0.0868632};
	std::complex<REAL> const weight_list[] = {0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04};
	std::complex<REAL> C_n = std::complex<REAL>(0.f, 0.f);
	std::complex<REAL> j = std::complex<REAL> (0.f, 1);



	// generate data
	std::vector< REAL > data(n_points_per_fit * n_fits);
	// Loop over all TEs
	for (size_t i = 0; i != data.size(); i++)
	{
		//size_t j = i / n_points_per_fit; // the fit
		size_t k = i % n_points_per_fit; // the position within a fit

		// C_n calculation
		C_n = std::complex<REAL>(0.f, 0.f);
		//Loop over all ppm/weight factors
		for (int h =0; h < 7; h++)
		{
			// weight_list * e ^ (j * 2 * pi * ppm_list * TEn)
			C_n += weight_list[h] * pow(expC, (j * 2.0f * piC * ppm_list[h] * TEnC[k]));
		}

		REAL x = user_info[k];
		std::cout << "Complex Number: " << C_n << "\n";
		REAL y = abs(true_parameters[0] * pow(exp, (-1 * true_parameters[2] * x)) + C_n * true_parameters[1] * pow(exp, (-1 * true_parameters[3] * x)));
		float rician_noise = sqrt(pow(normal_dist(rng),2) + pow(normal_dist(rng),2));
		data[i] = y + rician_noise;
		std::cout << "y             " << (REAL) y << "\n";
		std::cout << "rician noise  " << (REAL) rician_noise << "\n";
		std::cout << "y with noise  " << (REAL) data[i] << "\n";
	}

	// tolerance
	REAL const tolerance = 10e-3f;

	// maximum number of iterations
	int const max_number_iterations = 200;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

//error occurs if in models.cuh the parameters are wrong
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
// not sure how much from this area needs to be fixed
	// compute mean fitted parameters for converged fits
	std::vector< REAL > output_parameters_mean(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
// I"m pretty sure this needs to be edited for exponential functions
		if (output_states[i] == FitState::CONVERGED)
		{
			// add 1st parameter
			output_parameters_mean[0] += output_parameters[i * n_model_parameters + 0];
			// add 2nd parameter
			output_parameters_mean[1] += output_parameters[i * n_model_parameters + 1];
			// add 3rd parameter
			output_parameters_mean[2] += output_parameters[i * n_model_parameters + 2];
			// add 4th parameter
			output_parameters_mean[3] += output_parameters[i * n_model_parameters + 3];
		}
	}
	output_parameters_mean[0] /= output_states_histogram[0];
	output_parameters_mean[1] /= output_states_histogram[0];
	output_parameters_mean[2] /= output_states_histogram[0];
	output_parameters_mean[3] /= output_states_histogram[0];

	// compute std of fitted parameters for converged fits
	std::vector< REAL > output_parameters_std(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			// add squared deviation for 1st parameter
			output_parameters_std[0] += (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]) * (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]);
			// add squared deviation for 2nd parameter
			output_parameters_std[1] += (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]) * (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]);
			// add squared deviation for 3rd parameter
			output_parameters_std[2] += (output_parameters[i * n_model_parameters + 2] - output_parameters_mean[2]) * (output_parameters[i * n_model_parameters + 2] - output_parameters_mean[2]);
			// add squared deviation for 4th parameter
			output_parameters_std[3] += (output_parameters[i * n_model_parameters + 3] - output_parameters_mean[3]) * (output_parameters[i * n_model_parameters + 3] - output_parameters_mean[3]);
		}
	}
	// divide and take square root
	output_parameters_std[0] = sqrt(output_parameters_std[0] / output_states_histogram[0]);
	output_parameters_std[1] = sqrt(output_parameters_std[1] / output_states_histogram[0]);
	output_parameters_std[2] = sqrt(output_parameters_std[2] / output_states_histogram[0]);
	output_parameters_std[3] = sqrt(output_parameters_std[3] / output_states_histogram[0]);

	// print mean and std
	std::cout << "parameter 0   true " << true_parameters[0] << " mean " << output_parameters_mean[0] << " std " << output_parameters_std[0] << "\n";
	std::cout << "parameter 1   true " << true_parameters[1] << " mean " << output_parameters_mean[1] << " std " << output_parameters_std[1] << "\n";
	std::cout << "parameter 2   true " << true_parameters[2] << " mean " << output_parameters_mean[2] << " std " << output_parameters_std[2] << "\n";
	std::cout << "parameter 3   true " << true_parameters[3] << " mean " << output_parameters_mean[3] << " std " << output_parameters_std[3] << "\n";

	// compute mean chi-square for those converged
	REAL  output_chi_square_mean = 0;
	for (size_t i = 0; i != n_fits; i++)
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
	for (size_t i = 0; i != n_fits; i++)
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
	// liver_fat_three();
	liver_fat_four();

    std::cout << std::endl << "Example completed!" << std::endl;
    std::cout << "Press ENTER to exit" << std::endl;
    std::getchar();
	
	return 0;
}
