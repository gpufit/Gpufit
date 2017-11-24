#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>

BOOST_AUTO_TEST_CASE( Linear_Fit_1D )
{
	/*
		Performs a single fit using the Linear Fit (LINEAR_1D) model.
		- Uses user info
		- Uses trivial weights.
		- No noise is added.
		- Checks fitted parameters equalling the true parameters.
	*/

    std::size_t const n_fits = 1 ;
    std::size_t const n_points = 2 ;

	std::vector< float > const true_parameters{ 1, 1 };

    std::vector< float > data{ 1, 2 } ;

	std::vector< float > weights{ 1, 1 } ;

    std::vector< float > initial_parameters{ 1, 0 } ;

    float tolerance = 0.00001f;

	int max_n_iterations = 10 ;

	std::vector< int> parameters_to_fit{ 1, 1 } ;

	std::vector< float> user_info{ 0.f, 1.f } ;

	//std::array< float, 2 > output_parameters ;
    //int output_states ;
    //float output_chi_squares ;
    //int output_n_iterations ;
    //float output_data;
    std::vector< float > output_parameters(2);
    std::vector< int > output_states(1);
    std::vector< float > output_chi_square(1);
    std::vector< int > output_n_iterations(1);
    std::vector< float > output_data(2);

	// test with LSE
    int status_1 = gpufit
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
            n_points * sizeof( float ),
            reinterpret_cast< char * >( user_info.data() ),
            //output_parameters.data(),
            //&output_states,
            //&output_chi_squares,
            //&output_n_iterations,
            //&output_data
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_n_iterations.data(),
            output_data.data()
        ) ;
/*
    std::cout << output_parameters[0] <<" | " << true_parameters[0] <<"\n";
    std::cout << output_parameters[1] <<" | " << true_parameters[1] <<"\n";
    std::cout << output_states[0]<<"\n";
    std::cout << output_chi_square[0]<<"\n";
    std::cout << output_n_iterations[0] <<" | " << max_n_iterations <<"\n";
*/
    BOOST_CHECK( status_1 == 0 ) ;
	BOOST_CHECK( output_states[0] == 0 );
	BOOST_CHECK( output_n_iterations[0] <= max_n_iterations );
	BOOST_CHECK( output_chi_square[0] < 1e-6f );

	BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-6f);
	BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6f);

	// test with MLE
	int status_2 = gpufit
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
			n_points * sizeof(float),
			reinterpret_cast< char * >(user_info.data()),
			//output_parameters.data(),
            //&output_states,
            //&output_chi_squares,
            //&output_n_iterations,
            //&output_data
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_n_iterations.data(),
            output_data.data()
		);
/*
    std::cout << output_parameters[0] <<" | " << true_parameters[0] <<"\n";
    std::cout << output_parameters[1] <<" | " << true_parameters[1] <<"\n";
    std::cout << output_states[0]<<"\n";
    std::cout << output_chi_square[0]<<"\n";
    std::cout << output_n_iterations[0] <<" | " << max_n_iterations <<"\n";
*/
	BOOST_CHECK(status_2 == 0);
	BOOST_CHECK(output_states[0] == 0);
	BOOST_CHECK(output_n_iterations[0] <= max_n_iterations);
	BOOST_CHECK(output_chi_square[0] < 1e-6f);

	BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-6f);
	BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6f);

}
