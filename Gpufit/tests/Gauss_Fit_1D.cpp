#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>

template<std::size_t n_points>
void generate_gauss_1d(
    std::array< float, n_points >& values,
    std::array< float, n_points >& x_data,
    std::array< float, 4 > const & parameters )
{
    float const a = parameters[ 0 ];
    float const x0 = parameters[ 1 ];
    float const s = parameters[ 2 ];
    float const b = parameters[ 3 ];

    for ( int point_index = 0; point_index < n_points; point_index++ )
    {
        float const x = x_data[point_index];
        float const argx = ( ( x - x0 )*( x - x0 ) ) / ( 2.f * s * s );
        float const ex = exp( -argx );
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

    std::array< float, 4 > const true_parameters{ { 4.f, 2.f, 0.5f, 1.f } };

    std::array< float, n_points > x_data{ { 0.f, 1.f, 2.f, 3.f, 4.f} };
    std::array< float, n_points > data{};
    generate_gauss_1d(data, x_data, true_parameters);

    std::array< float, 4 > initial_parameters{ { 2.f, 1.5f, 0.3f, 0.f } };

    float tolerance{ 0.001f };

    int max_n_iterations{ 10 };

    std::array< int, 4 > parameters_to_fit{ { 1, 1, 1, 1 } };

    std::array< float, 4 > output_parameters;
    int output_states;
    float output_chi_square;
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
                &output_n_iterations,
                0
            ) ;

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_states == 0);
    BOOST_CHECK(output_chi_square < 1e-6f);
    BOOST_CHECK(output_n_iterations <= max_n_iterations);

    BOOST_CHECK(std::fabsf(output_parameters[0] - true_parameters[0]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[1] - true_parameters[1]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[2] - true_parameters[2]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[3] - true_parameters[3]) < 1e-6f);
}

void gauss_fit_1d_custom_x()
{
    /*
    Performs two fits using the GAUSS_1D model.
    - Doesn't use or weights.
    - Uses user_info for custom x coordinate values.
    - No noise is added.
    - Checks fitted parameters equalling the true parameters.
    - Compares fitted parameters of both fits
    */

    std::size_t const n_fits{ 2 };
    std::size_t const n_points{ 5 };

    std::array< float, 4 > const true_parameters{ { 4.f, 0.f, 0.25f, 1.f } };

    std::array< float, n_points > x_data = { { -1.f, -.5f, 0.f, .5f, 1.f } };
    std::array< float, n_points > one_fit_data{};
    generate_gauss_1d(one_fit_data, x_data, true_parameters);
    std::array< float, n_points * n_fits> data{};
    
    for (int i = 0; i < n_points; i++)
    {
        data[i] = one_fit_data[i];
        data[n_points + i] = one_fit_data[i];
    }

    std::array< float, 4 * n_fits > initial_parameters{ { 2.f, .25f, 0.15f, 0.f,
                                                          2.f, .25f, 0.15f, 0.f} };

    float tolerance{ 0.001f };

    int max_n_iterations{ 10 };

    std::array< int, 4 > parameters_to_fit{ { 1, 1, 1, 1 } };

    std::array< float, 4 * n_fits > output_parameters;
    std::array< int, n_fits > output_states;
    std::array< float, n_fits > output_chi_square;
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
            n_points * sizeof(float),
            reinterpret_cast< char * >(x_data.data()),
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_n_iterations.data(),
            0
        );
    // check gpufit status
    BOOST_CHECK(status == 0);
    
    // check first fit
    BOOST_CHECK(output_states[0] == 0);
    BOOST_CHECK(output_chi_square[0] < 1e-6f);
    BOOST_CHECK(output_n_iterations[0] <= max_n_iterations);

    BOOST_CHECK(std::fabsf(output_parameters[0] - true_parameters[0]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[1] - true_parameters[1]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[2] - true_parameters[2]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[3] - true_parameters[3]) < 1e-6f);

    // check second fit
    BOOST_CHECK(output_states[1] == 0);
    BOOST_CHECK(output_chi_square[1] < 1e-6f);
    BOOST_CHECK(output_n_iterations[1] <= max_n_iterations);

    BOOST_CHECK(std::fabsf(output_parameters[4] - true_parameters[0]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[5] - true_parameters[1]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[6] - true_parameters[2]) < 1e-6f);
    BOOST_CHECK(std::fabsf(output_parameters[7] - true_parameters[3]) < 1e-6f);

    // compare rusults of both fits
    BOOST_CHECK(output_states[0] == output_states[1]);
    BOOST_CHECK(output_chi_square[0] == output_chi_square[1]);
    BOOST_CHECK(output_n_iterations[0] == output_n_iterations[1]);

    BOOST_CHECK(output_parameters[0] == output_parameters[4]);
    BOOST_CHECK(output_parameters[1] == output_parameters[5]);
    BOOST_CHECK(output_parameters[2] == output_parameters[6]);
    BOOST_CHECK(output_parameters[3] == output_parameters[7]);
}

BOOST_AUTO_TEST_CASE( Gauss_Fit_1D )
{
    // single 1d gauss fit
    gauss_fit_1d();

    // two gauss fits with custom x coordinate values
    gauss_fit_1d_custom_x();
}
