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

#ifdef USE_CUBLAS
#define PRINT_IF_USE_CUBLAS() \
        std::cout << "CUBLAS enabled: Yes" << std::endl << std::endl
#else
#define PRINT_IF_USE_CUBLAS() \
        std::cout << "CUBLAS enabled: No" << std::endl << std::endl
#endif // USE_CUBLAS



/*
    Names of parameters for the 2D Gaussian peak model
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
void generate_test_parameters(std::vector<Parameters> & target,    Parameters const source)
{
    std::size_t const n_fits = target.size();

    std::uniform_real_distribution< REAL> uniform_dist(0, 1);

    REAL const a = 0.9f;
    REAL const b = 0.2f;

    int const text_width = 30;
    int const progress_width = 25;

    std::cout << std::setw(text_width) << " ";
    for (int i = 0; i < progress_width; i++)
        std::cout << "-";
    std::cout << std::endl;
    std::cout << std::setw(text_width) << std::left << "Generating test parameters";

    std::size_t progress = 0;

    for (std::size_t i = 0; i < n_fits; i++)
    {
        target[i].amplitude = source.amplitude * (a + b * uniform_dist(rng));
        target[i].center_x = source.center_x * (a + b * uniform_dist(rng));
        target[i].center_y = source.center_y * (a + b * uniform_dist(rng));
        target[i].width = source.width * (a + b * uniform_dist(rng));
        target[i].background = source.background * (a + b * uniform_dist(rng));

        progress += 1;
        if (progress >= n_fits / progress_width)
        {
            progress = 0;
            std::cout << "|";
        }
    }

    std::cout << std::endl;
    std::cout << std::setw(text_width) << " ";
    for (int i = 0; i < progress_width; i++)
        std::cout << "-";
    std::cout << std::endl;
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

    int const text_width = 30;
    int const progress_width = 25;

    std::cout << std::setw(text_width) << " ";
    for (int i = 0; i < progress_width; i++)
        std::cout << "-";
    std::cout << std::endl;
    std::cout << std::setw(text_width) << std::left << "Adding noise";

    std::size_t progress = 0;

    for (std::size_t i = 0; i < vec.size(); i++)
    {
        vec[i] += distribution(rng);

        progress += 1;
        if (progress >= vec.size() / progress_width)
        {
            progress = 0;
            std::cout << "|";
        }
    }

    std::cout << std::endl;
    std::cout << std::setw(text_width) << " ";
    for (int i = 0; i < progress_width; i++)
        std::cout << "-";
    std::cout << std::endl;
}

/*

*/
void generate_gauss2d(
    std::size_t const n_fits,
    std::size_t const n_points,
    std::vector<REAL> & data,
    std::vector<Parameters> const & parameters)
{
    int const text_width = 30;
    int const progress_width = 25;

    std::cout << std::setw(text_width) << " ";
    for (int i = 0; i < progress_width; i++)
        std::cout << "-";
    std::cout << std::endl;
    std::cout << std::setw(text_width) << std::left << "Generating data";

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
        if (progress >= n_fits / progress_width)
        {
            progress = 0;
            std::cout << "|";
        }
    }
    std::cout << std::endl;
    std::cout << std::setw(text_width) << " ";
    for (int i = 0; i < progress_width; i++)
        std::cout << "-";
    std::cout << std::endl;
}

/*
Runs Gpufit vs. Cpufit for various number of fits and compares the speed

No weights, Model: Gauss_2D, Estimator: LSE
*/
int main(int argc, char * argv[])
{
    // title 
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance comparison Gpufit vs. Cpufit" << std::endl;
    std::cout << "----------------------------------------" << std::endl << std::endl;

    std::cout << "Please note that execution speed test results depend on" << std::endl;
    std::cout << "the details of the CPU and GPU hardware." << std::endl;
    std::cout << std::endl;


    // check for CUDA availability
	bool const cuda_available = gpufit_cuda_available() != 0;
	if (!gpufit_cuda_available())
	{
		std::cout << "CUDA not available" << std::endl;
	}

	// check for CUDA runtime and driver
    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;
    bool const version_available = gpufit_get_cuda_version(&cuda_runtime_version, &cuda_driver_version) == ReturnState::OK;
    int const cuda_runtime_major = cuda_runtime_version / 1000;
    int const cuda_runtime_minor = cuda_runtime_version % 1000 / 10;
    int const cuda_driver_major = cuda_driver_version / 1000;
    int const cuda_driver_minor = cuda_driver_version % 1000 / 10;

    bool do_gpufits = false;
    if (cuda_available & version_available)
    {
        std::cout << "CUDA runtime version: ";
        std::cout << cuda_runtime_major << "." << cuda_runtime_minor << std::endl;
        std::cout << "CUDA driver version:  ";
        std::cout << cuda_driver_major << "." << cuda_driver_minor << std::endl;
        std::cout << std::endl;

        bool const cuda_available = cuda_driver_version > 0;
        if (cuda_available)
        {
            bool const version_compatible
                = cuda_driver_version >= cuda_runtime_version
                && cuda_runtime_version > 0;
            if (version_compatible)
            {
                do_gpufits = true;
            }
            else
            {
                std::cout << "The CUDA runtime version is not compatible with the" << std::endl;
                std::cout << "current graphics driver. Please update the driver, or" << std::endl;
                std::cout << "re - build Gpufit from source using a compatible version" << std::endl;
                std::cout << "of the CUDA toolkit." << std::endl;
                std::cout << std::endl;
            }
        }
        else
        {
            std::cout << "No CUDA enabled graphics card detected." << std::endl;
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "CUDA error detected. Error string: ";
        std::cout << gpufit_get_last_error() << std::endl;
        std::cout << std::endl;
    }
    if (!do_gpufits)
    {
        std::cout << "Skipping Gpufit computations." << std::endl << std::endl;
    }
    else
    {
        PRINT_IF_USE_CUBLAS();
    }

    // all numbers of fits
    std::vector<std::size_t> n_fits_all;
    if (sizeof(void*) < 8)
    {
        n_fits_all = { 10, 100, 1000, 10000, 100000, 1000000};
    }
    else
    {
        n_fits_all = { 10, 100, 1000, 10000, 100000, 1000000, 10000000 };
    }

    std::size_t const max_n_fits = n_fits_all.back();

    // fit parameters constant for every run
    std::size_t const size_x = 5;
    std::size_t const n_points = size_x * size_x;
    std::size_t const n_parameters = 5;
    std::vector<int> parameters_to_fit(n_parameters, 1);
    REAL const tolerance = 0.0001f;
    int const max_n_iterations = 10;

    // initial parameters
    Parameters true_parameters;
    true_parameters.amplitude = 500.;
    true_parameters.center_x = static_cast<REAL>(size_x) / 2 - 0.5f;
    true_parameters.center_y = static_cast<REAL>(size_x) / 2 - 0.5f;
    true_parameters.width = 1;
    true_parameters.background = 10.;

    // test parameters
    std::vector<Parameters> test_parameters(max_n_fits);
    generate_test_parameters(test_parameters, true_parameters);

    //  test data
    std::vector<REAL> data(max_n_fits * n_points);
    generate_gauss2d(max_n_fits, n_points, data, test_parameters);
    add_gauss_noise(data, true_parameters, 10);

    // initial parameter set
    std::vector<REAL> initial_parameters(n_parameters * max_n_fits);
    generate_initial_parameters(initial_parameters, test_parameters);

    // print collumn identifiers
    std::cout << std::endl << std::right;
    std::cout << std::setw(8) << "Number" << std::setw(3) << "|";
    std::cout << std::setw(13) << "Cpufit speed" << std::setw(3) << "|";
    std::cout << std::setw(13) << "Gpufit speed" << std::setw(3) << "|";
    std::cout << std::setw(12) << "Performance";
    std::cout << std::endl;
    std::cout << std::setw(8) << "of fits" << std::setw(3) << "|";
    std::cout << std::setw(13) << "(fits/s)" << std::setw(3) << "|";
    std::cout << std::setw(13) << "(fits/s)" << std::setw(3) << "|";
    std::cout << std::setw(12) << "gain factor";
    std::cout << std::endl;
    std::cout << "-------------------------------------------------------";
    std::cout << std::endl;

    // loop over number of fits
    for (std::size_t fit_index = 0; fit_index < n_fits_all.size(); fit_index++)
    {
        // number of fits
        std::size_t n_fits = n_fits_all[fit_index];
        std::cout << std::setw(8) << n_fits << std::setw(3) << "|";

        // Cpufit output
        std::vector<REAL> cpufit_parameters(n_fits * n_parameters);
        std::vector<int> cpufit_states(n_fits);
        std::vector<REAL> cpufit_chi_squares(n_fits);
        std::vector<int> cpufit_n_iterations(n_fits);

        // run Cpufit and measure time
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
        int const cpu_status
            = cpufit
            (
                n_fits,
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

        std::chrono::milliseconds::rep dt_gpufit = 0;

        // if we do not do gpufit, we skip the rest of the loop
        if (do_gpufits)
        {
            // Gpufit output parameters
            std::vector<REAL> gpufit_parameters(n_fits * n_parameters);
            std::vector<int> gpufit_states(n_fits);
            std::vector<REAL> gpufit_chi_squares(n_fits);
            std::vector<int> gpufit_n_iterations(n_fits);

            // run Gpufit and measure time
            t0 = std::chrono::high_resolution_clock::now();
            int const gpu_status
                = gpufit
                (
                n_fits,
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
            dt_gpufit = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count();

            if (gpu_status != 0)
            {
                // error in gpufit
                std::cout << "Error in gpufit: " << gpufit_get_last_error() << std::endl;
                do_gpufits = false;
            }
        }

        // print the calculation speed in fits/s
        std::cout << std::fixed << std::setprecision(0);
        if (dt_cpufit)
        {
            std::cout << std::setw(13) << static_cast<double>(n_fits) / static_cast<double>(dt_cpufit)* 1000.0 << std::setw(3) << "|";
        }
        else
        {
            std::cout << std::setw(13) << "inf" << std::setw(3) << "|";
        }
        if (dt_gpufit)
        {
            std::cout << std::setw(13) << static_cast<double>(n_fits) / static_cast<double>(dt_gpufit)* 1000.0 << std::setw(3) << "|";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(12) << static_cast<double>(dt_cpufit) / static_cast<double>(dt_gpufit);
        }
        else if (!do_gpufits)
        {
            std::cout << std::setw(13) << "--" << std::setw(3) << "|";
            std::cout << std::setw(12) << "--";
        }
        else
        {
            std::cout << std::setw(13) << "inf" << std::setw(3) << "|";
            std::cout << std::setw(12) << "inf";
        }
        
        std::cout << std::endl;        
    }
    std::cout << std::endl << "Test completed!" << std::endl;
    std::cout << "Press ENTER to exit" << std::endl;
    std::getchar();

    return 0;
}
