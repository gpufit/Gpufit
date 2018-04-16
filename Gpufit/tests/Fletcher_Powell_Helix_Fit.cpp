#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>

BOOST_AUTO_TEST_CASE( Fletcher_Powell_Helix )
{
    /*
        Performs a single fit using the FLETCHER_POWELL_HELIX model.
        - zero data is passed in.
        - Checks final chi_square to be near by zero.
        - Checks fitted parameters equalling the true parameters.
    */

    std::size_t const n_fits{ 1 } ;
    std::size_t const n_points{ 3 } ;
    std::size_t const n_parameters{ 3 } ;

    std::array< REAL, n_parameters > const true_parameters{ { 1., 0., 0. } };

    std::array< REAL, n_points > data{ { 0., 0., 0. } } ;

    std::array< REAL, n_parameters > initial_parameters{ { -1., 0., 0. } } ;

    REAL tolerance{ 1e-8f } ;
    
    int max_n_iterations{ 50 } ;
    
    std::array< int, n_parameters > parameters_to_fit{ { 1, 1, 1 } } ;
    
    std::array< REAL, n_parameters > output_parameters ;
    int output_state ;
    REAL output_chi_square ;
    int output_n_iterations ;

    // test initial_parameters * 1.
    int status = gpufit
        (
            n_fits,
            n_points,
            data.data(),
            0,
            FLETCHER_POWELL_HELIX,
            initial_parameters.data(),
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            LSE,
            0,
            0,
            output_parameters.data(),
            & output_state,
            & output_chi_square,
            & output_n_iterations
        ) ;

    BOOST_CHECK( status == 0 ) ;
    BOOST_CHECK( output_state == 0 );
    BOOST_CHECK( output_n_iterations <= 9 );
    BOOST_CHECK( output_chi_square < 1e-26f );

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-13);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-13);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-13);

    // test initial_parameters * 10.
    for (int i = 0; i < n_parameters; i++)
        initial_parameters[i] *= 10.;

    status = gpufit
    (
        n_fits,
        n_points,
        data.data(),
        0,
        FLETCHER_POWELL_HELIX,
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
    BOOST_CHECK(output_n_iterations <= 21);
    BOOST_CHECK(output_chi_square < 1e-26);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-13);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-13);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-13);

    // test initial_parameters * 100.
    for (int i = 0; i < n_parameters; i++)
        initial_parameters[i] *= 10.;

    status = gpufit
    (
        n_fits,
        n_points,
        data.data(),
        0,
        FLETCHER_POWELL_HELIX,
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
    BOOST_CHECK(output_n_iterations <= 29);
    BOOST_CHECK(output_chi_square < 1e-20);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-10);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-10);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-10);
}
