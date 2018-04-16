#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>
#include <cmath>

BOOST_AUTO_TEST_CASE( Linear_Fit_1D )
{
	/*
		Performs a single fit using the Linear Fit (LINEAR_1D) model.
		- Uses user info 
		- Uses trivial weights.
		- No noise is added.
		- Checks fitted parameters equalling the true parameters.
	*/

    std::size_t const n_fits{ 1 } ;
    std::size_t const n_points{ 2 } ;

	std::array< REAL, 2 > const true_parameters{ { 1, 1 } };

    std::array< REAL, n_points > data{ { 1, 2 } } ;
    
	std::array< REAL, n_points > weights{ { 1, 1 } } ;

    std::array< REAL, 2 > initial_parameters{ { 1, 0 } } ;

    REAL tolerance{ 0.00001f } ;
    
	int max_n_iterations{ 10 } ;
    
	std::array< int, 2 > parameters_to_fit{ { 1, 1 } } ;
    
	std::array< REAL, n_points > user_info{ { 0., 1. } } ;
    
	std::array< REAL, 2 > output_parameters ;
    int output_states ;
    REAL output_chi_squares ;
    int output_n_iterations ;

	// test with LSE
    int status = gpufit
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
            n_points * sizeof( REAL ),
            reinterpret_cast< char * >( user_info.data() ),
            output_parameters.data(),
            & output_states,
            & output_chi_squares,
            & output_n_iterations
        ) ;

    BOOST_CHECK( status == 0 ) ;
	BOOST_CHECK( output_states == 0 );
	BOOST_CHECK( output_n_iterations <= max_n_iterations );
	BOOST_CHECK( output_chi_squares < 1e-6f );

	BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-6);
	BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6);

	// test with MLE
	status = gpufit
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
			MLE,
			n_points * sizeof(REAL),
			reinterpret_cast< char * >(user_info.data()),
			output_parameters.data(),
			&output_states,
			&output_chi_squares,
			&output_n_iterations
		);

	BOOST_CHECK(status == 0);
	BOOST_CHECK(output_states == 0);
	BOOST_CHECK(output_n_iterations <= max_n_iterations);
	BOOST_CHECK(output_chi_squares < 1e-6);

	BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-6);
	BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6);

}
