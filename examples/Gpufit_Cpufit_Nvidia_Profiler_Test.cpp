/*
 * Runs 100k fits on the CPU and 2m fits on the GPU, used with the Nvidia profiler to obtain
 * running time information on the different CUDA kernels.
 */

#include "Cpufit/cpufit.h"
#include "Gpufit/gpufit.h"
#include "tests/utils.h"

#include <stdexcept>
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include <chrono>
#include <string>

#define _USE_MATH_DEFINES
#include <math.h>


/*
	Names of paramters for the 2D Gaussian peak model
*/
struct Parameters
{
	float amplitude;
	float center_x;
	float center_y;
	float width;
	float background;
};

/*
Prints some statistics and the speed (fits/second) of a run.
*/
void print_result(
    std::string const name,
    std::vector<float> const & estimated_parameters,
    std::vector<Parameters> const & test_parameters,
    std::vector<int> states,
    std::vector<int> const & n_iterations,
    std::size_t const n_fits,
    std::size_t const n_parameters,
    std::chrono::milliseconds::rep const duration_in_ms)
{

    std::vector<float> estimated_x_centers(n_fits);
    std::vector<float> test_x_centers(n_fits);

    for (std::size_t i = 0; i < n_fits; i++)
    {
        estimated_x_centers[i] = estimated_parameters[i*n_parameters + 1];
        test_x_centers[i] = test_parameters[i].center_x;
    }

    double const std_dev_x = calculate_standard_deviation(estimated_x_centers, test_x_centers, states);

    double const mean_n_iterations = calculate_mean(n_iterations, states);

    double fits_per_second = static_cast<double>(n_fits) / duration_in_ms * 1000;

    // output
    std::cout << std::fixed;

    std::cout << std::setw(5) << std::endl << "***" << name << "***";

    std::cout << std::setprecision(3);
    std::cout << std::setw(12) << duration_in_ms / 1000.0 << " s  ";

    std::cout << std::setprecision(2);
    std::cout << std::setw(12) << fits_per_second << " fits/s" << std::endl;

    std::cout << std::setprecision(6);
    std::cout << "x precision: " << std_dev_x << " px  ";

    std::cout << std::setprecision(2);
    std::cout << "mean iterations: " << mean_n_iterations << std::endl;
}

/*
Randomize parameters, slightly differently
*/
void generate_initial_parameters(std::vector<float> & parameters_set, std::vector<Parameters> const & parameters)
{
    std::uniform_real_distribution< float> uniform_dist(0, 1);

    float const a = 0.9f;
    float const b = 0.2f;

    int const n_parameters = sizeof(Parameters) / sizeof(float);
    for (std::size_t i = 0; i < parameters_set.size() / n_parameters; i++)
    {
        parameters_set[0 + i * n_parameters] = parameters[i].amplitude * (a + b * uniform_dist(rng));
        parameters_set[1 + i * n_parameters] = parameters[i].center_x * (a + b * uniform_dist(rng));
        parameters_set[2 + i * n_parameters] = parameters[i].center_y * (a + b * uniform_dist(rng));
        parameters_set[3 + i * n_parameters] = parameters[i].width * (a + b * uniform_dist(rng));
        parameters_set[4 + i * n_parameters] = parameters[i].background * (a + b * uniform_dist(rng));
    }
}

/*
Randomize parameters
*/
void generate_test_parameters(std::vector<Parameters> & target, Parameters const source)
{
    std::size_t const n_fits = target.size();

    std::uniform_real_distribution< float> uniform_dist(0, 1);

    float const a = 0.9f;
    float const b = 0.2f;

    for (std::size_t i = 0; i < n_fits; i++)
    {
        target[i].amplitude = source.amplitude * (a + b * uniform_dist(rng));
        target[i].center_x = source.center_x * (a + b * uniform_dist(rng));
        target[i].center_y = source.center_y * (a + b * uniform_dist(rng));
        target[i].width = source.width * (a + b * uniform_dist(rng));
        target[i].background = source.background * (a + b * uniform_dist(rng));
    }
}

/*

*/
void add_gauss_noise(std::vector<float> & vec, Parameters const & parameters, float const snr)
{
    float const gauss_fwtm = 4.292f * parameters.width; //only valid for circular gaussian
    float const fit_area = gauss_fwtm*gauss_fwtm;

    float const mean_amplitude = 2.f * float(M_PI) * parameters.amplitude * parameters.width * parameters.width / fit_area;

    float const std_dev = mean_amplitude / snr;

    std::normal_distribution<float> distribution(0.0, std_dev);

    for (std::size_t i = 0; i < vec.size(); i++)
    {
        vec[i] += distribution(rng);
    }
}

/*

*/
void generate_gauss2d(
	std::size_t const n_fits,
	std::size_t const n_points,
	std::vector<float> & data,
	std::vector<Parameters> const & parameters)
{
	std::cout << "generating " << n_fits << " fits ..." << std::endl;
	for (int i = 0; i < 50; i++)
		std::cout << "-";
	std::cout << std::endl;
	std::size_t progress = 0;

	for (std::size_t i = 0; i < n_fits; i++)
	{
		float const amplitude = parameters[i].amplitude;
		float const x00 = parameters[i].center_x;
		float const y00 = parameters[i].center_y;
		float const width = parameters[i].width;
		float const background = parameters[i].background;

		std::size_t const fit_index = i * n_points;

		for (int iy = 0; iy < sqrt(n_points); iy++)
		{
			for (int ix = 0; ix < sqrt(n_points); ix++)
			{
				std::size_t const point_index = iy * std::size_t(sqrt(n_points)) + ix;
				std::size_t const absolute_index = fit_index + point_index;

				float const argx
					= exp(-0.5f * ((ix - x00) / width) * ((ix - x00) / width));
				float const argy
					= exp(-0.5f * ((iy - y00) / width) * ((iy - y00) / width));

				data[absolute_index] = amplitude * argx * argy + background;
			}
		}

		progress += 1;
		if (progress >= n_fits / 50)
		{
			progress = 0;
			std::cout << "|";
		}
	}
	std::cout << std::endl;
	for (int i = 0; i < 50; i++)
		std::cout << "-";
	std::cout << std::endl;
}

/*
Runs Gpufit vs. Cpufit for various number of fits and compares the speed

No weights, Model: Gauss_2D, Estimator: LSE
*/
int main(int argc, char * argv[])
{
	// check for CUDA availability
	if (!gpufit_cuda_available())
	{
		std::cout << "CUDA not available" << std::endl;
		return -1;
	}

	// all numbers of fits
	std::size_t const n_fits_gpu = 2000000;
    std::size_t const n_fits_cpu = 100000;
	std::size_t const size_x = 15;
	std::size_t const n_points = size_x * size_x;

	// fit parameters constant for every run
	std::size_t const n_parameters = 5;
	std::vector<int> parameters_to_fit(n_parameters, 1);
	float const tolerance = 0.0001f;
	int const max_n_iterations = 10;

	// initial parameters
	Parameters true_parameters;
	true_parameters.amplitude = 500.f;
	true_parameters.center_x = static_cast<float>(size_x) / 2.f - 0.5f;
	true_parameters.center_y = static_cast<float>(size_x) / 2.f - 0.5f;
	true_parameters.width = 2.f;
	true_parameters.background = 10.f;

	//  test parameters
	std::cout << "generate test parameters" << std::endl;
	std::vector<Parameters> test_parameters(n_fits_gpu);
	generate_test_parameters(test_parameters, true_parameters);

	//  test data
	std::vector<float> data(n_fits_gpu * n_points);
	generate_gauss2d(n_fits_gpu, n_points, data, test_parameters);
	std::cout << "add noise" << std::endl;
	add_gauss_noise(data, true_parameters, 10.f);

	// initial parameter set
	std::vector<float> initial_parameters(n_parameters * n_fits_gpu);
	generate_initial_parameters(initial_parameters, test_parameters);

	std::cout << std::endl;
	std::cout << n_fits_cpu << " fits on the CPU" << std::endl;

	// Cpufit output
	std::vector<float> cpufit_parameters(n_fits_cpu * n_parameters);
	std::vector<int> cpufit_states(n_fits_cpu);
	std::vector<float> cpufit_chi_squares(n_fits_cpu);
	std::vector<int> cpufit_n_iterations(n_fits_cpu);

	// run Cpufit and measure time
	std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
	int const cpu_status
		= cpufit
		(
			n_fits_cpu,
			n_points,
			data.data(),
			0,
			GAUSS_2D,
			initial_parameters.data(),
			tolerance,
			max_n_iterations,
			parameters_to_fit.data(),
			LSE,
			0,
			0,
			cpufit_parameters.data(),
			cpufit_states.data(),
			cpufit_chi_squares.data(),
			cpufit_n_iterations.data()
		);
	std::chrono::milliseconds::rep const dt_cpufit = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count();

	if (cpu_status != 0)
	{
		// error in cpufit, should actually not happen
		std::cout << "Error in cpufit: " << cpufit_get_last_error() << std::endl;
	}
	else
	{
		// print
		print_result("Cpufit", cpufit_parameters, test_parameters, cpufit_states, cpufit_n_iterations, n_fits_cpu, n_parameters, dt_cpufit);
	}

    std::cout << std::endl;
    std::cout << n_fits_gpu << " fits on the GPU" << std::endl;

	// Gpufit output parameters
	std::vector<float> gpufit_parameters(n_fits_gpu * n_parameters);
	std::vector<int> gpufit_states(n_fits_gpu);
	std::vector<float> gpufit_chi_squares(n_fits_gpu);
	std::vector<int> gpufit_n_iterations(n_fits_gpu);

	// run Gpufit and measure time
	t0 = std::chrono::high_resolution_clock::now();
	int const gpu_status
		= gpufit
		(
			n_fits_gpu,
			n_points,
			data.data(),
			0,
			GAUSS_2D,
			initial_parameters.data(),
			tolerance,
			max_n_iterations,
			parameters_to_fit.data(),
			LSE,
			0,
			0,
			gpufit_parameters.data(),
			gpufit_states.data(),
			gpufit_chi_squares.data(),
			gpufit_n_iterations.data()
		);
	std::chrono::milliseconds::rep const dt_gpufit = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count();

	if (gpu_status != 0)
	{
		// error in gpufit
		std::cout << "Error in gpufit: " << gpufit_get_last_error() << std::endl;
	}
	else
	{
		// print results
		print_result("Gpufit", gpufit_parameters, test_parameters, gpufit_states, gpufit_n_iterations, n_fits_gpu, n_parameters, dt_gpufit);
	}

    std::cout << "\nPERFORMANCE GAIN Gpufit/Cpufit \t" << std::setw(10) << static_cast<double>(dt_cpufit) / dt_gpufit * n_fits_gpu / n_fits_cpu << std::endl;

	return 0;
}