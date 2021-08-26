#include "../gpufit.h"

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <numeric>
#include <math.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK_STATUS( cuda_function_call ) \
    if (cudaError_t const status = cuda_function_call) \
    { \
        throw std::runtime_error( cudaGetErrorString( status ) ) ; \
    }

template<class T>
struct GPU_array
{
    GPU_array(std::size_t const size)
    {
        CUDA_CHECK_STATUS(cudaMalloc(&data_, size * sizeof(T)));
    }

    GPU_array(std::vector<T> const & cpu_data) : GPU_array(cpu_data.size())
    {
        write(cpu_data);
    }

    GPU_array(std::size_t const & count, T const value) : GPU_array(count)
    {
        set(count, value);
    }

    ~GPU_array() { cudaFree(data_); }

    operator T * () { return static_cast<T *>(data_); }

    void read(std::vector<T> & to) const
    {
        CUDA_CHECK_STATUS(cudaMemcpy(
            to.data(), data_, to.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void write(std::vector<T> const & from)
    {
        CUDA_CHECK_STATUS(cudaMemcpy(
            data_, from.data(), from.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    void set(std::size_t const count, T const value)
    {
        CUDA_CHECK_STATUS(cudaMemset(data_, 1, count * sizeof(T)));
    }

private:
    void * data_;
};

void generate_gauss_2d(
    std::vector<REAL> const & x_coordinates,
    std::vector<REAL> const & y_coordinates,
    std::vector<REAL> const & gauss_params, 
    std::vector<REAL> & output_values)
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
        
        REAL arg = -(   (x_coordinates[i] - gauss_params[1]) * (x_coordinates[i] - gauss_params[1]) 
                       + (y_coordinates[i] - gauss_params[2]) * (y_coordinates[i] - gauss_params[2])   ) 
                     / (2 * gauss_params[3] * gauss_params[3]);
                     
        output_values[i] = gauss_params[0] * exp(arg) + gauss_params[4];
        
    }
}

void cuda_interface_example()
{
    /*
        This example generates test data on the CPU in form of 10000 two dimensional
        Gaussian peaks with the size of 50x50 data points per peak. It is noised by Poisson
        distributed noise. The initial guesses were randomized, within a specified range
        of the true value. Before call to Gpufit the input data is transfered to GPU memory.
        The GAUSS_2D model is fitted to the test data sets using the MLE estimator. After
        calling Gpufit the output data is transfered to CPU memory.

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
    size_t const size_x = 20;
    size_t const n_points_per_fit = size_x * size_x;
    size_t const n_parameters = 5;

    // true parameters (amplitude, center x position, center y position, width, offset)
    std::vector< REAL > true_parameters{ 5, 14.5f, 14.5f, 3, 10}; 
    
    std::cout << "generate example data" << std::endl;

    // initialize random number generator
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution< REAL> uniform_dist(0, 1);

    // initial parameters (randomized)
    std::vector< REAL > initial_parameters(n_fits * n_parameters);
    for (size_t i = 0; i < n_fits; i++)
    {
        for (size_t j = 0; j < n_parameters; j++)
        {
            if (j == 1 || j == 2)
            {
                initial_parameters[i * n_parameters + j]
                    = true_parameters[j] + true_parameters[3] 
                    * (-.2f + .4f * uniform_dist(rng));
            }
            else
            {
                initial_parameters[i * n_parameters + j]
                    = true_parameters[j] * (.8f + .4f * uniform_dist(rng));
            }
        }
    }

    // generate x and y values
    std::vector< REAL > x(n_points_per_fit);
    std::vector< REAL > y(n_points_per_fit);
    for (size_t i = 0; i < size_x; i++)
    {
        for (size_t j = 0; j < size_x; j++) {
            x[i * size_x + j] = static_cast<REAL>(j);
            y[i * size_x + j] = static_cast<REAL>(i);
        }
    }

    // generate test data with Poisson noise
    std::vector< REAL > temp(n_points_per_fit);
    generate_gauss_2d(x, y, true_parameters, temp);

    std::vector< REAL > data(n_fits * n_points_per_fit);
    for (size_t i = 0; i < n_fits; i++)
    {
        for (size_t j = 0; j < n_points_per_fit; j++)
        {
            std::poisson_distribution< int > poisson_dist(temp[j]);
            data[i * n_points_per_fit + j] = static_cast<REAL>(poisson_dist(rng));
        }
    }

    // tolerance
    REAL const tolerance = .001f;

    // maximum number of iterations
    int const max_n_iterations = 20;

    // estimator ID
    int const estimator_id = MLE;

    // model ID
    int const model_id = GAUSS_2D;

    // parameters to fit (all of them)
    std::vector< int > parameters_to_fit(n_parameters, 1);

    // output parameters CPU
    std::vector< REAL > output_parameters(n_fits * n_parameters);
    std::vector< int > output_states(n_fits);
    std::vector< REAL > output_chi_squares(n_fits);
    std::vector< int > output_n_iterations(n_fits);

    // input parameters GPU
    GPU_array<REAL> gpu_data(data);
    GPU_array<REAL> gpu_weights(data.size(), 1);

    // input/output parameters GPU
    GPU_array<REAL> gpu_initial_parameters(initial_parameters);

    // output_parameters GPU
    GPU_array<int> gpu_states(n_fits);
    GPU_array<REAL> gpu_chi_squares(n_fits);
    GPU_array<int> gpu_n_iterations(n_fits);

    // call to gpufit_cuda_interface
    std::chrono::high_resolution_clock::time_point time_0 = std::chrono::high_resolution_clock::now();
    int status = gpufit_cuda_interface
        (
            n_fits,
            n_points_per_fit,
            gpu_data,
            gpu_weights,
            model_id,
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            estimator_id,
            0,
            0,
            gpu_initial_parameters,
            gpu_states,
            gpu_chi_squares,
            gpu_n_iterations
        );
    std::chrono::high_resolution_clock::time_point time_1 = std::chrono::high_resolution_clock::now();

    // check status
    if (status != ReturnState::OK)
    {
        throw std::runtime_error(gpufit_get_last_error());
    }

    // copy output data to CPU memory
    gpu_initial_parameters.read(output_parameters);
    gpu_states.read(output_states);
    gpu_chi_squares.read(output_chi_squares);
    gpu_n_iterations.read(output_n_iterations);

    std::cout << std::endl << "unconstrained fit" << std::endl;
    
    // print execution time
    std::cout << "execution time "
        << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count() << " ms" << std::endl;

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

    // compute mean of fitted parameters for converged fits
    std::vector< REAL > output_parameters_mean(n_parameters, 0);
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == FitState::CONVERGED)
        {
            for (size_t j = 0; j < n_parameters; j++)
            {
                output_parameters_mean[j] += output_parameters[i * n_parameters + j];
            }
        }
    }
    // normalize
    for (size_t j = 0; j < n_parameters; j++)
    {
        output_parameters_mean[j] /= output_states_histogram[0];
    }
    
    // compute std of fitted parameters for converged fits
    std::vector< REAL > output_parameters_std(n_parameters, 0);
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == FitState::CONVERGED)
        {
            for (size_t j = 0; j < n_parameters; j++)
            {
                output_parameters_std[j]
                    += (output_parameters[i * n_parameters + j] - output_parameters_mean[j])
                    *  (output_parameters[i * n_parameters + j] - output_parameters_mean[j]);
            }
        }
    }
    // normalize and take square root
    for (size_t j = 0; j < n_parameters; j++)
    {
        output_parameters_std[j] = sqrt(output_parameters_std[j] / output_states_histogram[0]);
    }

    // print true value, fitted mean and std for every parameter
    for (size_t j = 0; j < n_parameters; j++)
    {
        std::cout
            << "parameter "     << j
            << " true "         << true_parameters[j]
            << " fitted mean "  << output_parameters_mean[j]
            << " std "          << output_parameters_std[j] << std::endl;
    }

    // compute mean chi-square for those converged
    REAL  output_chi_square_mean = 0;
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == FitState::CONVERGED)
        {
            output_chi_square_mean += output_chi_squares[i];
        }
    }
    output_chi_square_mean /= static_cast<REAL>(output_states_histogram[0]);
    std::cout << "mean chi square " << output_chi_square_mean << std::endl;

    // compute mean number of iterations for those converged
    REAL  output_number_iterations_mean = 0;
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == FitState::CONVERGED)
        {
            output_number_iterations_mean += static_cast<REAL>(output_n_iterations[i]);
        }
    }
    // normalize
    output_number_iterations_mean /= static_cast<REAL>(output_states_histogram[0]);
    std::cout << "mean number of iterations " << output_number_iterations_mean << std::endl;

    // define constraints
    std::vector< REAL > constraints(n_fits * 2 * n_parameters, 0);
    for (size_t i = 0; i != n_fits; i++)
    {
        constraints[i * n_parameters * 2 + 6] = 2.9f;
        constraints[i * n_parameters * 2 + 7] = 3.1f;
    }
    GPU_array<REAL> gpu_constraints(constraints);
    std::vector< int > constraint_types(n_parameters, 0);
    constraint_types[0] = 1; // lower
    constraint_types[3] = 3; // lower and upper
    constraint_types[4] = 1; // lower
    
    // call to gpufit_constrained_cuda_interface
    time_0 = std::chrono::high_resolution_clock::now();
    status = gpufit_constrained_cuda_interface
    (
        n_fits,
        n_points_per_fit,
        gpu_data,
        gpu_weights,
        model_id,
        tolerance,
        max_n_iterations,
        parameters_to_fit.data(),
        gpu_constraints,
        constraint_types.data(),
        estimator_id,
        0,
        0,
        gpu_initial_parameters,
        gpu_states,
        gpu_chi_squares,
        gpu_n_iterations
    );
    time_1 = std::chrono::high_resolution_clock::now();

    // check status
    if (status != ReturnState::OK)
    {
        throw std::runtime_error(gpufit_get_last_error());
    }

    // copy output data to CPU memory
    gpu_initial_parameters.read(output_parameters);
    gpu_states.read(output_states);
    gpu_chi_squares.read(output_chi_squares);
    gpu_n_iterations.read(output_n_iterations);

    std::cout << std::endl << "constrained fit" << std::endl;

    // print execution time
    std::cout << "execution time "
        << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count() << " ms" << std::endl;

    // get fit states
    std::fill(output_states_histogram.begin(), output_states_histogram.end(), 0);
    for (std::vector< int >::iterator it = output_states.begin(); it != output_states.end(); ++it)
    {
        output_states_histogram[*it]++;
    }

    std::cout << "ratio converged              " << (REAL)output_states_histogram[0] / n_fits << "\n";
    std::cout << "ratio max iteration exceeded " << (REAL)output_states_histogram[1] / n_fits << "\n";
    std::cout << "ratio singular hessian       " << (REAL)output_states_histogram[2] / n_fits << "\n";
    std::cout << "ratio neg curvature MLE      " << (REAL)output_states_histogram[3] / n_fits << "\n";
    std::cout << "ratio gpu not read           " << (REAL)output_states_histogram[4] / n_fits << "\n";

    // compute mean of fitted parameters for converged fits
    std::fill(output_parameters_mean.begin(), output_parameters_mean.end(), 0.f);
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == FitState::CONVERGED)
        {
            for (size_t j = 0; j < n_parameters; j++)
            {
                output_parameters_mean[j] += output_parameters[i * n_parameters + j];
            }
        }
    }
    // normalize
    for (size_t j = 0; j < n_parameters; j++)
    {
        output_parameters_mean[j] /= output_states_histogram[0];
    }

    // compute std of fitted parameters for converged fits
    std::fill(output_parameters_std.begin(), output_parameters_std.end(), 0.f);
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == FitState::CONVERGED)
        {
            for (size_t j = 0; j < n_parameters; j++)
            {
                output_parameters_std[j]
                    += (output_parameters[i * n_parameters + j] - output_parameters_mean[j])
                    * (output_parameters[i * n_parameters + j] - output_parameters_mean[j]);
            }
        }
    }
    // normalize and take square root
    for (size_t j = 0; j < n_parameters; j++)
    {
        output_parameters_std[j] = sqrt(output_parameters_std[j] / output_states_histogram[0]);
    }

    // print true value, fitted mean and std for every parameter
    for (size_t j = 0; j < n_parameters; j++)
    {
        std::cout
            << "parameter " << j
            << " true " << true_parameters[j]
            << " fitted mean " << output_parameters_mean[j]
            << " std " << output_parameters_std[j] << std::endl;
    }

    // compute mean chi-square for those converged
    output_chi_square_mean = 0;
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == FitState::CONVERGED)
        {
            output_chi_square_mean += output_chi_squares[i];
        }
    }
    output_chi_square_mean /= static_cast<REAL>(output_states_histogram[0]);
    std::cout << "mean chi square " << output_chi_square_mean << std::endl;

    // compute mean number of iterations for those converged
    output_number_iterations_mean = 0;
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == FitState::CONVERGED)
        {
            output_number_iterations_mean += static_cast<REAL>(output_n_iterations[i]);
        }
    }
    // normalize
    output_number_iterations_mean /= static_cast<REAL>(output_states_histogram[0]);
    std::cout << "mean number of iterations " << output_number_iterations_mean << std::endl;

}

int main(int argc, char *argv[])
{
    cuda_interface_example();

    std::cout << std::endl << "Example completed!" << std::endl;
    std::cout << "Press ENTER to exit" << std::endl;
    std::getchar();

    return 0;
}
