#include "../../Gpufit/gpufit.h"

#include <time.h>
#include <vector>
#include <random>
#include <iostream>
#include <math.h>

void two_compartment_exchange_four()
{

	/*
	This example generates test data in form of 10000 one dimensional Tissue Concentration
	curves, using a synthetic AIF, hard coded Ktrans and vp values. Gaussian noise is
	added to achieve a specific SNR value. The initial guess is varied randomly between
	10% and 180% of the true value. The same x position values are used for
	every fit.

	The console output shows
	- the ratio of converged fits including ratios of not converged fits for
	  different reasons,
	- the SNR of the generated data
	- the values of the true parameters and the mean values of the fitted
	  parameters including their standard deviation,
	- the mean chi square value
	- and the mean number of iterations needed.
	*/

	// start timer
	clock_t time_start, time_end;
	time_start = clock();


	// number of fits, fit points and parameters
	size_t const n_fits = 100000;
	size_t const n_points_per_fit = 60;
	size_t const n_model_parameters = 4;
	REAL snr = 4.8;

	// true parameters
	std::vector< REAL > true_parameters{ 0.005, 0.3, 0.05, 0.1 };		// Ktrans, ve, vp, Fp

	// custom x positions for the data points of every fit, stored in user info
	// time independent variable, given in minutes
	REAL timeX[] ={ 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5,
					5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10,
					10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25,
					14.5, 14.75, 15 };

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
	//rng.seed(0);
	std::uniform_real_distribution< REAL > uniform_dist(0, 1);
	std::normal_distribution< REAL > normal_dist(0, 1);

	// initial parameters (randomized)
	std::vector< REAL > initial_parameters(n_fits * n_model_parameters);
	for (size_t i = 0; i != n_fits; i++)
	{
		// random Ktrans
		initial_parameters[i * n_model_parameters + 0] = true_parameters[0] * (0.95f + 0.1f * uniform_dist(rng));
		// random ve
		initial_parameters[i * n_model_parameters + 1] = true_parameters[1] * (0.95f + 0.1f * uniform_dist(rng));
		// random vp
		initial_parameters[i * n_model_parameters + 2] = true_parameters[2] * (0.95f + 0.1f * uniform_dist(rng));
		// random Fp
		initial_parameters[i * n_model_parameters + 3] = true_parameters[3] * (0.95f + 0.1f * uniform_dist(rng));
	}

	// parameter_constraints
	std::vector< REAL > parameter_constraints(n_fits * n_model_parameters * 2);
	std::vector< int > constraint_type(n_fits * n_model_parameters);
	for (size_t i = 0; i != n_fits; i++)
	{
		// Ktrans
		parameter_constraints[i * n_model_parameters * 2 + 0] = 0;
		parameter_constraints[i * n_model_parameters * 2 + 1] = 2;
		// ve
		parameter_constraints[i * n_model_parameters * 2 + 2] = 0.02;
		parameter_constraints[i * n_model_parameters * 2 + 3] = 1;
		// vp
		parameter_constraints[i * n_model_parameters * 2 + 4] = 0.001;
		parameter_constraints[i * n_model_parameters * 2 + 5] = 1;
		// Fp
		parameter_constraints[i * n_model_parameters * 2 + 6] = 0.001;
		parameter_constraints[i * n_model_parameters * 2 + 7] = 100;

		//type 3=upper lower
		constraint_type[i * n_model_parameters + 0] = 3;
		constraint_type[i * n_model_parameters + 1] = 3;
		constraint_type[i * n_model_parameters + 2] = 3;
		constraint_type[i * n_model_parameters + 3] = 3;
	}

	// generate data
	std::vector< REAL > data(n_points_per_fit * n_fits);
	REAL mean_y = 0;
	for (size_t i = 0; i != data.size(); i++)
	{
		size_t j = i / n_points_per_fit; // the fit
		size_t k = i % n_points_per_fit; // the position within a fit

		REAL conv = 0;
		REAL Tp = true_parameters[2] / (true_parameters[3] / ((true_parameters[3] / true_parameters[0]) - 1) + true_parameters[3]);
		REAL Te = true_parameters[1] / (true_parameters[3] / ((true_parameters[3] / true_parameters[0]) - 1));
		REAL Tb = true_parameters[2] / true_parameters[3];
		REAL Kpos = 0.5 * (1/Tp + 1/Te + sqrt(pow(1/Tp + 1/Te,2) - 4 * 1/Te * 1/Tb));
		REAL Kneg = 0.5 * (1/Tp + 1/Te - sqrt(pow(1/Tp + 1/Te,2) - 4 * 1/Te * 1/Tb));
		REAL Eneg = (Kpos - 1/Tb) / (Kpos - Kneg);
		for (int n = 1; n < k; n++) {

			REAL spacing = timeX[n] - timeX[n - 1];
			REAL Ct = Cp[n] * (exp(-(timeX[k] - timeX[n]) * Kpos) + Eneg * (exp(-(timeX[k] - timeX[n]) * Kneg) - exp(-Kpos)));
			REAL Ctprev = Cp[n - 1] * (exp(-(timeX[k] - timeX[n-1]) * Kpos) + Eneg * ( exp(-(timeX[k] - timeX[n-1]) * Kneg) - exp(-Kpos)));
			conv += ((Ct + Ctprev) / 2 * spacing);
		}
		REAL y = true_parameters[3] * conv;
		data[i] = y;
		mean_y += y;
//		std::cout << data[i] << std::endl;
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
	int const model_id = TWO_COMPARTMENT_EXCHANGE;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

	// output parameters
	std::vector< REAL > output_parameters(n_fits * n_model_parameters);
	std::vector< int > output_states(n_fits);
	std::vector< REAL > output_chi_square(n_fits);
	std::vector< int > output_number_iterations(n_fits);

	// call to gpufit (C interface)
	int const status = gpufit_constrained
	(
		n_fits,
		n_points_per_fit,
		data.data(),
		0,
		model_id,
		initial_parameters.data(),
		parameter_constraints.data(),
		constraint_type.data(),
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
			// add Fp
			output_parameters_mean[2] += output_parameters[i * n_model_parameters + 2];
			// add Fp
			output_parameters_mean[3] += output_parameters[i * n_model_parameters + 3];
			// add Ktrans
			output_parameters_mean_error[0] += abs(output_parameters[i * n_model_parameters + 0]-true_parameters[0]);
			// add vp
			output_parameters_mean_error[1] += abs(output_parameters[i * n_model_parameters + 1]-true_parameters[1]);
			// add Fp
			output_parameters_mean_error[2] += abs(output_parameters[i * n_model_parameters + 2]-true_parameters[2]);
			// add Fp
			output_parameters_mean_error[3] += abs(output_parameters[i * n_model_parameters + 3]-true_parameters[3]);

			if (output_parameters[i * n_model_parameters + 1]<0)
			{
				//std::cout << "Ktrans  fit " << output_parameters[i * n_model_parameters + 0]  << " error " << abs(output_parameters[i * n_model_parameters + 0]-true_parameters[0]) << "\n";
				std::cout << "ve	fit " << output_parameters[i * n_model_parameters + 1]  << " error " << abs(output_parameters[i * n_model_parameters + 1]-true_parameters[1]) << "\n";
				//std::cout << "vp	fit " << output_parameters[i * n_model_parameters + 2]  << " error " << abs(output_parameters[i * n_model_parameters + 2]-true_parameters[2]) << "\n";
				//std::cout << "Fp	fit " << output_parameters[i * n_model_parameters + 2]  << " error " << abs(output_parameters[i * n_model_parameters + 2]-true_parameters[2]) << "\n";

				std::cout << "Ktrans  init " << initial_parameters[i * n_model_parameters + 0] << "\n";
				std::cout << "ve	init " << initial_parameters[i * n_model_parameters + 1] << "\n";
				std::cout << "vp	init " << initial_parameters[i * n_model_parameters + 2] << "\n";
				std::cout << "Fp	init " << initial_parameters[i * n_model_parameters + 3] << "\n";
			}

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
			// add squared deviation for Ktrans
			output_parameters_std[0] += (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]) * (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]);
			// add squared deviation for vp
			output_parameters_std[1] += (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]) * (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]);
			// add squared deviation for Fp
			output_parameters_std[2] += (output_parameters[i * n_model_parameters + 2] - output_parameters_mean[2]) * (output_parameters[i * n_model_parameters + 2] - output_parameters_mean[2]);
			// add squared deviation for Fp
			output_parameters_std[3] += (output_parameters[i * n_model_parameters + 3] - output_parameters_mean[3]) * (output_parameters[i * n_model_parameters + 3] - output_parameters_mean[3]);
		}
	}
	// divide and take square root
	output_parameters_std[0] = sqrt(output_parameters_std[0] / output_states_histogram[0]);
	output_parameters_std[1] = sqrt(output_parameters_std[1] / output_states_histogram[0]);
	output_parameters_std[2] = sqrt(output_parameters_std[2] / output_states_histogram[0]);
	output_parameters_std[3] = sqrt(output_parameters_std[3] / output_states_histogram[0]);


	// print mean and std
	std::cout << "Data SNR:  " << snr << "\n";
	std::cout << "Ktrans  true " << true_parameters[0] << " mean " << output_parameters_mean[0] << " std " << output_parameters_std[0] << "\n";
	std::cout << "ve	true " << true_parameters[1] << " mean " << output_parameters_mean[1] << " std " << output_parameters_std[1] << "\n";
	std::cout << "vp	true " << true_parameters[2] << " mean " << output_parameters_mean[2] << " std " << output_parameters_std[2] << "\n";
	std::cout << "Fp	true " << true_parameters[3] << " mean " << output_parameters_mean[3] << " std " << output_parameters_std[3] << "\n";


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
	std::cout << std::endl << "Beginning two compartment exchange fit..." << std::endl;
	two_compartment_exchange_four();

	std::cout << std::endl << "Two compartment exchange fit completed!" << std::endl;

	return 0;
}
