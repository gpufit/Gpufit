#include "../gpufit.h"

#include <time.h>
#include <vector>
#include <random>
#include <iostream>
#include <math.h>

void patlak_two()
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

	// start timer
	clock_t time_start, time_end;
	time_start = clock();


	// number of fits, fit points and parameters
	size_t const n_fits = 10000;
	size_t const n_points_per_fit = 60;
	size_t const n_model_parameters = 2;
	REAL snr = 0.8;

	// custom x positions for the data points of every fit, stored in user info
	// time independent variable, given in minutes
	REAL timeX[] = { 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5,
					5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10,
					10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15 };

	// Concentration of plasma (independent), at 1 min based on equation: Cp(t) = 5.5e^(-.6t)
	REAL Cp[] =   {	0.0f, 0.0f, 0.0f, 3.01846399851715f, 2.59801604007558f, 2.2361331285733f, 1.92465762011135f, 1.65656816551711f, 1.4258214335524f,
					1.22721588081636f, 1.05627449741415f, 0.909143885218726f, 0.782507393725825f, 0.6735103553914f, 0.579695735090254f,
					0.498948743091769f, 0.429449163006342f, 0.369630320068624f, 0.318143764811612f, 0.273828876023252f, 0.235686697768721f,
					0.20285742070682f, 0.174601000079374f, 0.150280473460109f, 0.12934760220805f, 0.111330512951924f, 0.0958230605172143f,
					0.0824756725126274f, 0.0709874691926393f, 0.0610994809603327f, 0.0525888106179893f, 0.0452636087696102f, 0.0389587491097867f,
					0.033532106110336f, 0.0288613511954976f, 0.0248411951843697f, 0.0213810148391187f, 0.018402810016092f, 0.0158394453694853f,
					0.013633136971665f, 0.0117341497352074f, 0.010099676273659f, 0.00869287192804919f, 0.00748202420651342f, 0.00643983791435146f,
					0.00554281985976681f, 0.00477074926518851f, 0.00410622194607174f, 0.00353425798195557f, 0.00304196403581309f, 0.00261824270962248f, 
					0.00225354238438883f, 0.00193964190545541f, 0.00166946525943377f, 0.00143692206515917f, 0.00123677028298367f, 0.00106449804756952f,
					0.000916221960431984f, 0.000788599549519612f, 0.000678753922476738f };


	std::vector< REAL > user_info(2 * n_points_per_fit);
	for (size_t i = 0; i < n_points_per_fit; i++)
	{
		user_info[i] = static_cast<REAL>(timeX[i]);
	}

	for (size_t i = n_points_per_fit; i < 2 * n_points_per_fit; i++)
	{
		user_info[i] = static_cast<REAL>(Cp[i - n_points_per_fit]);
	}

	// size of user info in bytes
	size_t const user_info_size = 2 * n_points_per_fit * sizeof(REAL);

	// initialize random number generator
	std::mt19937 rng;
	rng.seed(time(NULL));
	std::uniform_real_distribution< REAL > uniform_dist(0, 1);
	std::normal_distribution< REAL > normal_dist(0, 1);

	// true parameters
	std::vector< REAL > true_parameters{ 0.05, 0.03 };		// Ktrans, vp

	// initial parameters (randomized)
	std::vector< REAL > initial_parameters(n_fits * n_model_parameters);
	for (size_t i = 0; i != n_fits; i++)
	{
		// random offset
		initial_parameters[i * n_model_parameters + 0] = true_parameters[0] * (0.1f + 1.8f * uniform_dist(rng));
		// random slope
		initial_parameters[i * n_model_parameters + 1] = true_parameters[0] * (0.1f + 1.8f * uniform_dist(rng));
	}

	// generate data
	std::vector< REAL > data(n_points_per_fit * n_fits);
	REAL mean_y = 0;
	for (size_t i = 0; i != data.size(); i++)
	{
		size_t j = i / n_points_per_fit; // the fit
		size_t k = i % n_points_per_fit; // the position within a fit
		REAL x = 0;
		for (int n = 1; n < k; n++) {
		
			REAL spacing = timeX[n] - timeX[n - 1];
			x += (Cp[n - 1] + Cp[n]) / 2 * spacing;
		}
		REAL y = true_parameters[0] * x + true_parameters[1] * Cp[k];
		//data[i] = y + normal_dist(rng);
		//data[i] = y * (0.2f + 1.6f * uniform_dist(rng));
		data[i] = y;
		mean_y += y;
		//std::cout << data[i] << std::endl;
	}
	mean_y = mean_y / data.size();
	std::normal_distribution<REAL> norm_snr(0,mean_y/snr);
	for (size_t i = 0; i != data.size(); i++)
	{
		data[i] = data[i] + norm_snr(rng);
	}



	// tolerance
	REAL const tolerance = 10e-8f;

	// maximum number of iterations
	int const max_number_iterations = 200;

	// estimator ID
	int const estimator_id = LSE;

	// model ID
	int const model_id = PATLAK;

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
		reinterpret_cast< char* >( user_info.data() ),
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

	std::cout << "ratio converged              " << (REAL)output_states_histogram[0] / n_fits << "\n";
	std::cout << "ratio max iteration exceeded " << (REAL)output_states_histogram[1] / n_fits << "\n";
	std::cout << "ratio singular hessian       " << (REAL)output_states_histogram[2] / n_fits << "\n";
	std::cout << "ratio neg curvature MLE      " << (REAL)output_states_histogram[3] / n_fits << "\n";
	std::cout << "ratio gpu not read           " << (REAL)output_states_histogram[4] / n_fits << "\n";

	// compute mean fitted parameters for converged fits
	std::vector< REAL > output_parameters_mean(n_model_parameters, 0);
	std::vector< REAL > output_parameters_mean_error(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			// add Ktrans
			output_parameters_mean[0] += output_parameters[i * n_model_parameters + 0];
			// add vp
			output_parameters_mean[1] += output_parameters[i * n_model_parameters + 1];
			// add Ktrans
			output_parameters_mean_error[0] += abs(output_parameters[i * n_model_parameters + 0]-true_parameters[0]);
			// add vp
			output_parameters_mean_error[1] += abs(output_parameters[i * n_model_parameters + 1]-true_parameters[1]);
		}
	}
	output_parameters_mean[0] /= output_states_histogram[0];
	output_parameters_mean[1] /= output_states_histogram[0];

	// compute std of fitted parameters for converged fits
	std::vector< REAL > output_parameters_std(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			// add squared deviation for Ktrans
			output_parameters_std[0] += (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]) * (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]);
			// add squared deviation for vp
			output_parameters_std[1] += (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]) * (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]);
		}
	}
	// divide and take square root
	output_parameters_std[0] = sqrt(output_parameters_std[0] / output_states_histogram[0]);
	output_parameters_std[1] = sqrt(output_parameters_std[1] / output_states_histogram[0]);

	// print mean and std
	std::cout << "Ktrans  true " << true_parameters[0] << " mean " << output_parameters_mean[0] << " std " << output_parameters_std[0] << "\n";
	std::cout << "vp	true " << true_parameters[1] << " mean " << output_parameters_mean[1] << " std " << output_parameters_std[1] << "\n";

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

	// time
	//time(&time_end);
	time_end = clock();
	double time_taken_sec = double(time_end-time_start)/double(CLOCKS_PER_SEC);
	std::cout << "execution time for " << n_fits << " fits was " << time_taken_sec << " seconds\n";
}


int main(int argc, char* argv[])
{
	std::cout << std::endl << "Beginning Patlak fit..." << std::endl;
	patlak_two();

	std::cout << std::endl << "Patlak fit completed!" << std::endl;

	return 0;
}
