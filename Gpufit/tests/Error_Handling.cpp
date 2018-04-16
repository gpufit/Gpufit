#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>

BOOST_AUTO_TEST_CASE( Error_Handling )
{
    std::size_t const n_fits{ 1 } ;
    std::size_t const n_points{ 2 } ;
    std::array< REAL, n_points > data{ { 0, 1 } } ;
    std::array< REAL, n_points > weights{ { 1, 1 } } ;
    std::array< REAL, 2 > initial_parameters{ { 0, 0 } } ;
    REAL tolerance{ 0.001f } ;
    int max_n_iterations{ 10 } ;
    std::array< int, 2 > parameters_to_fit{ { 0, 0 } } ;
    std::array< int, 2 > user_info{ { 0, 1 } } ;
    std::array< REAL, 2 > output_parameters ;
    int output_states ;
    REAL output_chi_square ;
    int output_n_iterations ;

    int const status
            = gpufit
            (
                n_fits,
                n_points,
                data.data(),
                weights.data(),
                LINEAR_1D,
                initial_parameters.data(),
                tolerance,
                max_n_iterations,
                parameters_to_fit.data(),
                LSE,
                n_points * sizeof( int ),
                reinterpret_cast< char * >( user_info.data() ),
                output_parameters.data(),
                & output_states,
                & output_chi_square,
                & output_n_iterations
            ) ;

    BOOST_CHECK( status == - 1 ) ;

    std::string const error = gpufit_get_last_error() ;

    BOOST_CHECK( error == "invalid configuration argument" ) ;
}
