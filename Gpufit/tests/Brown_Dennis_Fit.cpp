#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>
#include <cmath>
#include <cfloat>

BOOST_AUTO_TEST_CASE( Brown_Dennis )
{
    /*
    Performs a single fit using the FLETCHER_POWELL_HELIX model.
    - zero data is passed in.
    - Checks final chi_square to be smaller than the expected value.
    - Checks fitted parameters equalling the true parameters.
    */

    std::size_t const n_fits{ 1 };
    std::size_t const n_points{ 20 };
    std::size_t const n_parameters{ 4 };

    std::array< REAL, n_parameters > const true_parameters{ { -7.3027296f, 10.800486f, -0.33427560f, 0.13945465f } };
    
    REAL const expected_chi_square = 38821.5f;
    
    std::array< REAL, n_points > data;
    data.fill(0);

    std::array< REAL, n_parameters > initial_parameters{ { 25, 5, -5, 1 } };

    REAL tolerance{ 1e-8f };

    int max_n_iterations{ 1000 };

    std::array< int, n_parameters > parameters_to_fit{ { 1, 1, 1, 1 } };

    std::array< REAL, n_parameters > output_parameters;
    int output_state;
    REAL output_chi_square;
    int output_n_iterations;

    // test initial_parameters * 1
    int status = gpufit
    (
        n_fits,
        n_points,
        data.data(),
        0,
        BROWN_DENNIS,
        initial_parameters.data(),
        tolerance,
        max_n_iterations,
        parameters_to_fit.data(),
        LSE,
        0,
        0,
        output_parameters.data(),
        &output_state,
        &output_chi_square,
        &output_n_iterations
    );

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_state == 0);
    BOOST_CHECK(output_n_iterations <= 83);
    BOOST_CHECK(output_chi_square < expected_chi_square);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-2f);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-2f);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-2f);
    BOOST_CHECK(std::abs(output_parameters[3] - true_parameters[3]) < 1e-2f);

    // test initial_parameters * 10
    for (int i = 0; i < n_parameters; i++)
        initial_parameters[i] *= 10;

    status = gpufit
    (
        n_fits,
        n_points,
        data.data(),
        0,
        BROWN_DENNIS,
        initial_parameters.data(),
        tolerance,
        max_n_iterations,
        parameters_to_fit.data(),
        LSE,
        0,
        0,
        output_parameters.data(),
        &output_state,
        &output_chi_square,
        &output_n_iterations
    );

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_state == 0);
    BOOST_CHECK(output_n_iterations <= 191);
    BOOST_CHECK(output_chi_square < expected_chi_square);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-2f);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-2f);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-2f);
    BOOST_CHECK(std::abs(output_parameters[3] - true_parameters[3]) < 1e-2f);

    // test initial_parameters * 100
    for (int i = 0; i < n_parameters; i++)
        initial_parameters[i] *= 10;

    status = gpufit
    (
        n_fits,
        n_points,
        data.data(),
        0,
        BROWN_DENNIS,
        initial_parameters.data(),
        tolerance,
        max_n_iterations,
        parameters_to_fit.data(),
        LSE,
        0,
        0,
        output_parameters.data(),
        &output_state,
        &output_chi_square,
        &output_n_iterations
    );

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_state == 0);
    BOOST_CHECK(output_n_iterations <= 555);
    BOOST_CHECK(output_chi_square < expected_chi_square);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-1f);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-1f);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-1f);
    BOOST_CHECK(std::abs(output_parameters[3] - true_parameters[3]) < 1e-1f);
}
