#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>
#include <cmath>

template<std::size_t n_points, std::size_t n_parameters>
void generate_gauss_1d(
    std::array< REAL, n_points >& values,
    std::array< REAL, n_points >& x_data,
    std::array< REAL, n_parameters > const & parameters )
{
    REAL const a = parameters[ 0 ];
    REAL const x0 = parameters[ 1 ];
    REAL const s = parameters[ 2 ];
    REAL const b = parameters[ 3 ];

    for ( int point_index = 0; point_index < n_points; point_index++ )
    {
        REAL const x = x_data[point_index];
        REAL const argx = ( ( x - x0 )*( x - x0 ) ) / ( 2 * s * s );
        REAL const ex = exp( -argx );
        values[ point_index ] = a * ex + b;
    }
}

void gauss_fit_1d()
{
    /*
    Performs a single fit using the GAUSS_1D model.
    - Doesn't use user_info or weights.
    - No noise is added.
    - Checks fitted parameters equalling the true parameters.
    */

    std::size_t const n_fits{ 1 };
    std::size_t const n_points{ 5 };
    std::size_t const n_parameters{ 4 };

    std::array< REAL, n_parameters > const true_parameters{ { 4, 2, .5f, 1 } };

    std::array< REAL, n_points > x_data{ { 0, 1, 2, 3, 4} };
    std::array< REAL, n_points > data{};
    generate_gauss_1d(data, x_data, true_parameters);

    std::array< REAL, n_parameters > initial_parameters{ { 2, 1.5f, 0.3f, 0 } };

    REAL tolerance{ 0.001f };

    int max_n_iterations{ 10 };

    std::array< int, n_parameters > parameters_to_fit{ { 1, 1, 1, 1 } };

    std::array< REAL, n_parameters > output_parameters;
    int output_states;
    REAL output_chi_square;
    int output_n_iterations;

    int const status
        = gpufit
        (
            n_fits,
            n_points,
            data.data(),
            0,
            GAUSS_1D,
            initial_parameters.data(),
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            LSE,
            0,
            0,
            output_parameters.data(),
            &output_states,
            &output_chi_square,
            &output_n_iterations
        );

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_states == 0);
    BOOST_CHECK(output_chi_square < 1e-6f);
    BOOST_CHECK(output_n_iterations <= max_n_iterations);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[3] - true_parameters[3]) < 1e-6f);
}

void gauss_fit_1d_custom_x()
{
    /*
    Performs two fits using the GAUSS_1D model.
    - Doesn't use or weights.
    - Uses user_info for custom x coordinate values, unique for each fit.
    - No noise is added.
    - Checks fitted parameters equalling the true parameters.
    */

    std::size_t const n_fits{ 2 };
    std::size_t const n_points{ 5 };
    std::size_t const n_parameters{ 4 };

    std::array< REAL, n_parameters > const true_parameters_1{ { 4, 0, .25f, 1 } };
    std::array< REAL, n_parameters > const true_parameters_2{ { 6, .5f, .15f, 2 } };

    std::array< REAL, n_parameters > initial_parameters_1{ { 2, .25f, .15f, 0 } };
    std::array< REAL, n_parameters > initial_parameters_2{ { 8, .75f, .2f, 3 } };

    std::array< REAL, n_points > x_data_1 = { { -1, -.5f, 0, .5f, 1 } };
    std::array< REAL, n_points > x_data_2 = { { 0, .25f, .5f, .75f, 1 } };

    std::array< REAL, n_points > fit_data_1{};
    std::array< REAL, n_points > fit_data_2{};

    generate_gauss_1d(fit_data_1, x_data_1, true_parameters_1);
    generate_gauss_1d(fit_data_2, x_data_2, true_parameters_2);
    
    std::array< REAL, n_points * n_fits> data{};
    std::array< REAL, n_points * n_fits> x_data{};
    
    for (int i = 0; i < n_points; i++)
    {
        data[i] = fit_data_1[i];
        data[n_points + i] = fit_data_2[i];

        x_data[i] = x_data_1[i];
        x_data[n_points + i] = x_data_2[i];
    }

    std::array< REAL, n_parameters * n_fits> initial_parameters{};

    for (int i = 0; i < n_parameters; i++)
    {
        initial_parameters[i] = initial_parameters_1[i];
        initial_parameters[n_parameters + i] = initial_parameters_2[i];
    }

    REAL tolerance{ 1e-6f };

    int max_n_iterations{ 20 };

    std::array< int, n_parameters > parameters_to_fit{ { 1, 1, 1, 1 } };

    std::array< REAL, n_parameters * n_fits > output_parameters;
    std::array< int, n_fits > output_states;
    std::array< REAL, n_fits > output_chi_square;
    std::array< int, n_fits > output_n_iterations;

    int const status
        = gpufit
        (
            n_fits,
            n_points,
            data.data(),
            0,
            GAUSS_1D,
            initial_parameters.data(),
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            LSE,
            n_points * n_fits * sizeof(REAL),
            reinterpret_cast< char * >(x_data.data()),
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_n_iterations.data()
        );
    // check gpufit status
    BOOST_CHECK(status == 0);
    
    // check first fit
    BOOST_CHECK(output_states[0] == 0);
    BOOST_CHECK(output_chi_square[0] < 1e-6f);
    BOOST_CHECK(output_n_iterations[0] <= max_n_iterations);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters_1[0]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters_1[1]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters_1[2]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[3] - true_parameters_1[3]) < 1e-6f);

    // check second fit
    BOOST_CHECK(output_states[1] == 0);
    BOOST_CHECK(output_chi_square[1] < 1e-6f);
    BOOST_CHECK(output_n_iterations[1] <= max_n_iterations);

    BOOST_CHECK(std::abs(output_parameters[4] - true_parameters_2[0]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[5] - true_parameters_2[1]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[6] - true_parameters_2[2]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[7] - true_parameters_2[3]) < 1e-6f);
}

BOOST_AUTO_TEST_CASE( Gauss_Fit_1D )
{
    // single 1d gauss fit
    gauss_fit_1d();

    // two gauss fits with custom x coordinate values
    gauss_fit_1d_custom_x();
}
