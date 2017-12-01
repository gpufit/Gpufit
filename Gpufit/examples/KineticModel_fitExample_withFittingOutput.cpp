#include "../gpufit.h"

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <numeric>
#include <math.h>
#include "../constants.h"
#include <stdexcept>
#include <iomanip>


void kinetic_fit_example()
{
    /*
        This example generates test data in form of 2'000'000 one dimensional Positron
        Emission Tomography (PET) time-activity curve (TAC).
        It starts from a real TAC array and then it replicates it adding gaussian noise
        to have 2'000'000 slightly different curves to fit.
        The initial guesses were randomized, within a specified range of the true value,
    	where with true value we mean a reference output of the fitting computed with
    	another already validated software.
    	The BICOMP_3EXP_3K model is fitted to the test data sets using the LSE estimator.
        The console output shows
         - the execution time,
         - the ratio of converged fits including ratios of not converged fits for
           different reasons,
         - the values of the true parameters and the mean values of the fitted
           parameters including their standard deviation,
         - the mean chi square value
         - and the mean number of iterations needed.
    */


    // number of fits, fit points and parameters
    size_t const n_fits = 1000;
    std::cout << "number of parallel fits: "<< n_fits << "\n";

    size_t const n_points_per_fit = 24;
    size_t const n_model_parameters = 5;

    // initialize random number generator
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution< float > uniform_dist(0, 1);
    std::normal_distribution< float > normal_dist(0, 1);

    // additional input params via user_info
    std::vector< float > times_float =  {  0.08333333 ,0.25,0.41666666,
                                           0.58333334,0.75,0.91666666,
                                           1.08333333,1.25,1.41666667,
                                           1.58333333,1.75,1.91666667,
                                           2.25,2.75,3.5,4.5,5.5,7.,9.,
                                           12.5,17.5,22.5,27.5,35.
                                        };
    std::vector< float > IFvalue =  {   0.,0.,24409.38070751,29004.28711479,
                                        16902.29913917,12060.98208071,
                                        10612.57271934,10197.68263123,
                                        10054.6062693,9978.15052684,
                                        9917.65977008,9861.25459037,
                                        9698.43991838,9541.40325626,
                                        9243.27864281,8965.43931906,
                                        8706.77619679,8242.85724982,
                                        7843.85989999,7087.59694672,
                                        6609.51306567,6344.98718338,
                                        6246.22490223,6414.56761034
                                    };
    std::vector< float > IFparam =  { 0.458300896195880,
                                      758028.906510941,
                                      3356.00773871079,
                                      7042.64861309165,
                                     -9.91821801288336,
                                      0.0134846319687693,
                                     -0.0585800774301212
                                    };
    size_t const user_info_size = ( IFvalue.size() + IFparam.size() + times_float.size() ) * sizeof(float);
    std::vector< float > user_info;
    user_info.reserve(user_info_size/sizeof(float));
    user_info.insert(user_info.end(), IFparam.begin(), IFparam.end());
    user_info.insert(user_info.end(), IFvalue.begin(), IFvalue.end());
    user_info.insert(user_info.end(), times_float.begin(), times_float.end());

    // true parameters (amplitude, center x position, center y position, width, offset)
    std::vector< float > true_parameters = {0.22, 0.40,  0.55,   0.12,  0.014 };

    std::vector<float> data_meas = {0.,342.14285714,4250.07142857,
                                    8019.71428571, 7703.,8685.21428571,
                                    5077.57142857,4767.21428571,7816.71428571,
                                    8750.28571429,6621.35714286,9039.35714286,
                                    10013.64285714,7516.92857143,11464.57142857,
                                    12454.28571429,10803.85714286,11094.42857143,
                                    14108.85714286,14878.28571429,15847.78571429,
                                    17448.71428571,18997.85714286,19569.};

    std::vector<float> data(n_fits*n_points_per_fit);
    std::vector<float> init_parameters(n_fits * n_model_parameters);

    // generate parameter initializations
    for (size_t i = 0; i != init_parameters.size(); i++)
    {
        size_t j = i / n_model_parameters; // the param vector
        size_t k = i % n_model_parameters; // the position within a param vector
        init_parameters[i] = true_parameters[k] * (0.8f + 0.4f * uniform_dist(rng));
    }

    // generate data
    for (size_t i = 0; i != data.size(); i++)
    {
        size_t j = i / n_points_per_fit; // the fit
        size_t k = i % n_points_per_fit; // the position within a fit
        data[i] = data_meas[k] + 2*normal_dist(rng);
    }

    // tolerance
    float const tolerance = 0.001f;

    // maximum number of iterations
    int const max_number_iterations = 10;

    // estimator ID
    EstimatorID const estimator_id = LSE;

    // model ID
    ModelID const model_id = BICOMP_3EXP_3K;

    // parameters to fit (all of them)
    std::vector< int > parameters_to_fit(n_model_parameters, 1);

    // output parameters
    std::vector< float > output_parameters(n_fits * n_model_parameters);
    std::vector< int > output_states(n_fits);
    std::vector< float > output_chi_square(n_fits);
    std::vector< int > output_number_iterations(n_fits);
    std::vector< float > output_data(n_fits * n_points_per_fit);

    // call to gpufit (C interface)
    std::chrono::high_resolution_clock::time_point time_0 = std::chrono::high_resolution_clock::now();
    int const status = gpufit
        (
            n_fits,
            n_points_per_fit,
            data.data(),
            0,
            model_id,
            init_parameters.data(),
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
            output_data.data()
        );

    std::chrono::high_resolution_clock::time_point time_1 = std::chrono::high_resolution_clock::now();

    // check status
    if (status != ReturnState::OK)
    {
        std::cout << "-----------Runtime Error-----------\n";
        throw std::runtime_error(gpufit_get_last_error());
    }

    // print execution time
    std::cout
        << "execution time: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(time_1 - time_0).count() << " ms\n";

    // get fit states
    std::vector< int > output_states_histogram(5, 0);
    for (std::vector< int >::iterator it = output_states.begin(); it != output_states.end(); ++it)
    {
        output_states_histogram[*it]++;
    }

    std::cout << "ratio converged              " << (float)output_states_histogram[0] / n_fits * 100 << "%\n";
    std::cout << "ratio max iteration exceeded " << (float)output_states_histogram[1] / n_fits * 100 << "%\n";
    std::cout << "ratio singular hessian       " << (float)output_states_histogram[2] / n_fits * 100 << "%\n";
    std::cout << "ratio neg curvature MLE      " << (float)output_states_histogram[3] / n_fits * 100 << "%\n";
    std::cout << "ratio gpu not read           " << (float)output_states_histogram[4] / n_fits * 100 << "%\n";

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
            << " parameter "     << j
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

    for (int i = 0; i < n_points_per_fit; i++)
    {
        std::cout << "measure : " << std::fixed << std::setw( 8 ) <<  std::setprecision( 2 ) << std::setfill( ' ' )
                             << data_meas[i] << " || "
                  << "fitting : " << std::fixed << std::setw( 8 ) <<  std::setprecision( 2 ) << std::setfill( ' ' )
                             << output_data.data()[i]
                  << '\n';
    }

}

int main(int argc, char *argv[])
{
    kinetic_fit_example();

    std::cout << std::endl << "Example completed!" << std::endl;
    std::cout << "Press ENTER to exit" << std::endl;
    std::getchar();

    return 0;
}