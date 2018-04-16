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
	REAL amplitude;
	REAL center_x;
	REAL center_y;
	REAL width;
	REAL background;
};

/*
Prints some statistics and the speed (fits/second) of a run.
*/
void print_result(
    std::string const name,
    std::vector<REAL> const & estimated_parameters,
    std::vector<Parameters> const & test_parameters,
    std::vector<int> states,
    std::vector<int> const & n_iterations,
    std::size_t const n_fits,
    std::size_t const n_parameters,
    std::chrono::milliseconds::rep const duration_in_ms)
{

    std::vector<REAL> estimated_x_centers(n_fits);
    std::vector<REAL> test_x_centers(n_fits);

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
void generate_initial_parameters(std::vector<REAL> & parameters_set, std::vector<Parameters> const & parameters)
{
    std::uniform_real_distribution< REAL> uniform_dist(0, 1);

    REAL const a = 0.9f;
    REAL const b = 0.2f;

    int const n_parameters = sizeof(Parameters) / sizeof(REAL);
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

    std::uniform_real_distribution< REAL> uniform_dist(0, 1);

    REAL const a = 0.9f;
    REAL const b = 0.2f;

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
void add_gauss_noise(std::vector<REAL> & vec, Parameters const & parameters, REAL const snr)
{
    REAL const gauss_fwtm = 4.292f * parameters.width; //only valid for circular gaussian
    REAL const fit_area = gauss_fwtm*gauss_fwtm;

    REAL const mean_amplitude = 2 * REAL(M_PI) * parameters.amplitude * parameters.width * parameters.width / fit_area;

    REAL const std_dev = mean_amplitude / snr;

    std::normal_distribution<REAL> distribution(0, std_dev);

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
	std::vector<REAL> & data,
	std::vector<Parameters> const & parameters)
{
	std::cout << "generating " << n_fits << " fits ..." << std::endl;
	for (int i = 0; i < 50; i++)
		std::cout << "-";
	std::cout << std::endl;
	std::size_t progress = 0;

	for (std::size_t i = 0; i < n_fits; i++)
	{
		REAL const amplitude = parameters[i].amplitude;
		REAL const x00 = parameters[i].center_x;
		REAL const y00 = parameters[i].center_y;
		REAL const width = parameters[i].width;
		REAL const background = parameters[i].background;

		std::size_t const fit_index = i * n_points;

		for (int iy = 0; iy < sqrt(n_points); iy++)
		{
			for (int ix = 0; ix < sqrt(n_points); ix++)
			{
				std::size_t const point_index = iy * std::size_t(sqrt(n_points)) + ix;
				std::size_t const absolute_index = fit_index + point_index;

				REAL const argx
					= exp(-0.5f * ((ix - x00) / width) * ((ix - x00) / width));
				REAL const argy
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
    std::size_t n_fits_gpu;
    if (sizeof(void*) < 8)
    {
        n_fits_gpu = 1000000;
    }
    else
    {
        n_fits_gpu = 2000000;
    }
    std::size_t const n_fits_cpu = 100000;
	std::size_t const size_x = 15;
	std::size_t const n_points = size_x * size_x;

	// fit parameters constant for every run
	std::size_t const n_parameters = 5;
	std::vector<int> parameters_to_fit(n_parameters, 1);
	REAL const tolerance = .0001f;
	int const max_n_iterations = 10;

	// initial parameters
	Parameters true_parameters;
	true_parameters.amplitude = 500;
	true_parameters.center_x = static_cast<REAL>(size_x) / 2 - .5f;
	true_parameters.center_y = static_cast<REAL>(size_x) / 2 - .5f;
	true_parameters.width = 2;
	true_parameters.background = 10;

	//  test parameters
	std::cout << "generate test parameters" << std::endl;
	std::vector<Parameters> test_parameters(n_fits_gpu);
	generate_test_parameters(test_parameters, true_parameters);

	//  test data
	std::vector<REAL> data(n_fits_gpu * n_points);
	generate_gauss2d(n_fits_gpu, n_points, data, test_parameters);
	std::cout << "add noise" << std::endl;
	add_gauss_noise(data, true_parameters, 10);

	// initial parameter set
	std::vector<REAL> initial_parameters(n_parameters * n_fits_gpu);
	generate_initial_parameters(initial_parameters, test_parameters);

	std::cout << std::endl;
	std::cout << n_fits_cpu << " fits on the CPU" << std::endl;

	// Cpufit output
	std::vector<REAL> cpufit_parameters(n_fits_cpu * n_parameters);
	std::vector<int> cpufit_states(n_fits_cpu);
	std::vector<REAL> cpufit_chi_squares(n_fits_cpu);
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
	std::vector<REAL> gpufit_parameters(n_fits_gpu * n_parameters);
	std::vector<int> gpufit_states(n_fits_gpu);
	std::vector<REAL> gpufit_chi_squares(n_fits_gpu);
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
