#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>
#include <cmath>
#include <random>

BOOST_AUTO_TEST_CASE( Patlak )
{
	/*
		Performs a single fit using the Patlak (PATLAK) model.
		- Uses user info 
		- Uses trivial weights.
		- No noise is added.
		- Checks fitted parameters equalling the true parameters.
	*/

    std::size_t const n_fits{ 10000 } ;
    std::size_t const n_points{ 60 } ;
	REAL snr = 0.8f; 
	std::array< REAL, 2 > const true_parameters{ { .05, .03 } };
	// custom x positions for the data points of every fit, stored in user info
	// time independent variable, given in minutes
	const REAL timeX[] = { 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5,
					5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10,
					10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15 };

	// Concentration of plasma (independent), at 1 min based on equation: Cp(t) = 5.5e^(-.6t)
	const REAL Cp[] = {	0.0f, 0.0f, 0.0f, 3.01846399851715f, 2.59801604007558f, 2.2361331285733f, 1.92465762011135f, 1.65656816551711f, 1.4258214335524f,
					1.22721588081636f, 1.05627449741415f, 0.909143885218726f, 0.782507393725825f, 0.6735103553914f, 0.579695735090254f,
					0.498948743091769f, 0.429449163006342f, 0.369630320068624f, 0.318143764811612f, 0.273828876023252f, 0.235686697768721f,
					0.20285742070682f, 0.174601000079374f, 0.150280473460109f, 0.12934760220805f, 0.111330512951924f, 0.0958230605172143f,
					0.0824756725126274f, 0.0709874691926393f, 0.0610994809603327f, 0.0525888106179893f, 0.0452636087696102f, 0.0389587491097867f,
					0.033532106110336f, 0.0288613511954976f, 0.0248411951843697f, 0.0213810148391187f, 0.018402810016092f, 0.0158394453694853f,
					0.013633136971665f, 0.0117341497352074f, 0.010099676273659f, 0.00869287192804919f, 0.00748202420651342f, 0.00643983791435146f,
					0.00554281985976681f, 0.00477074926518851f, 0.00410622194607174f, 0.00353425798195557f, 0.00304196403581309f, 0.00261824270962248f, 
					0.00225354238438883f, 0.00193964190545541f, 0.00166946525943377f, 0.00143692206515917f, 0.00123677028298367f, 0.00106449804756952f,
					0.000916221960431984f, 0.000788599549519612f, 0.000678753922476738f };

    // std::array< REAL, n_points > data{ { 1, 2 } } ;
	std::array< REAL, n_points*n_fits > data;
	std::mt19937 rng;
	rng.seed(time(NULL));
	REAL mean_y = 0;
	for (size_t i = 0; i != data.size(); i++)
	{
		size_t j = i / n_points; // the fit
		size_t k = i % n_points; // the position within a fit
		REAL x = 0;
		for (int n = 1; n < k; n++) {
		
			REAL spacing = timeX[n] - timeX[n - 1];
			x += (Cp[n - 1] + Cp[n]) / 2 * spacing;
		}
		REAL y = true_parameters[0] * x + true_parameters[1] * Cp[k];
		//data[i] = y + normal_dist(rng);
		//data[i] = y * (0.2f + 1.6f * uniform_dist(rng));
		data[i] = y;
		mean_y += y;
		//std::cout << data[i] << std::endl;
	}
	mean_y = mean_y / data.size();
	std::normal_distribution<REAL> norm_snr(0,mean_y/snr);
	for (size_t i = 0; i < data.size(); i++)
	{
		data[i] = data[i] + norm_snr(rng);
	}
	// std::array< REAL, n_points > weights{ { 1, 1 } } ;

    // std::array< REAL, 2 > initial_parameters{ { 1, 0 } } ;
	// initial parameters (randomized)
	std::vector< REAL > initial_parameters(n_fits * 2);
	std::uniform_real_distribution< REAL > uniform_dist(0, 1);
	for (size_t i = 0; i < n_fits; i++)
	{
		// random offset
		initial_parameters[i * 2 + 0] = true_parameters[0] * (0.1f + 1.8f * uniform_dist(rng));
		// random slope
		initial_parameters[i * 2 + 1] = true_parameters[1] * (0.1f + 1.8f * uniform_dist(rng));
	}

    REAL tolerance{ 10e-8f } ;
    
	int max_n_iterations{ 200 } ;
    
	std::array< int, 2 > parameters_to_fit{ { 1, 1 } } ;

	std::array< REAL, n_points*2 > user_info{ {0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5,
					5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10,
					10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15,
					0.0f, 0.0f, 0.0f, 3.01846399851715f, 2.59801604007558f, 2.2361331285733f, 1.92465762011135f, 1.65656816551711f, 1.4258214335524f,
					1.22721588081636f, 1.05627449741415f, 0.909143885218726f, 0.782507393725825f, 0.6735103553914f, 0.579695735090254f,
					0.498948743091769f, 0.429449163006342f, 0.369630320068624f, 0.318143764811612f, 0.273828876023252f, 0.235686697768721f,
					0.20285742070682f, 0.174601000079374f, 0.150280473460109f, 0.12934760220805f, 0.111330512951924f, 0.0958230605172143f,
					0.0824756725126274f, 0.0709874691926393f, 0.0610994809603327f, 0.0525888106179893f, 0.0452636087696102f, 0.0389587491097867f,
					0.033532106110336f, 0.0288613511954976f, 0.0248411951843697f, 0.0213810148391187f, 0.018402810016092f, 0.0158394453694853f,
					0.013633136971665f, 0.0117341497352074f, 0.010099676273659f, 0.00869287192804919f, 0.00748202420651342f, 0.00643983791435146f,
					0.00554281985976681f, 0.00477074926518851f, 0.00410622194607174f, 0.00353425798195557f, 0.00304196403581309f, 0.00261824270962248f, 
					0.00225354238438883f, 0.00193964190545541f, 0.00166946525943377f, 0.00143692206515917f, 0.00123677028298367f, 0.00106449804756952f,
					0.000916221960431984f, 0.000788599549519612f, 0.000678753922476738f} } ;
    
	std::array< REAL, n_fits * 2 > output_parameters ;
	std::vector< int > output_states(n_fits);
	std::vector< REAL > output_chi_squares(n_fits);
	std::vector< int > output_n_iterations(n_fits);

	// test with LSE
    int status = gpufit
        (
            n_fits,
            n_points,
            data.data(),
			0,
            // weights.data(),
            PATLAK,
            initial_parameters.data(),
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            LSE,
            n_points * sizeof( REAL ) * 2 ,
            reinterpret_cast< char * >( user_info.data() ),
            output_parameters.data(),
            output_states.data(),
            output_chi_squares.data(),
            output_n_iterations.data()
        ) ;

    BOOST_CHECK( status == 0 ) ;
	// BOOST_CHECK( output_states == 0 );
	// BOOST_CHECK( output_n_iterations <= max_n_iterations );
	// BOOST_CHECK( output_chi_squares < 1e-6f );
	std::cout << "XD: " << output_parameters[0] << " " << true_parameters[0];
	std::cout << "XD: " << output_parameters[1] << " " << true_parameters[1];
	BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-6);
	BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6);

	// test with MLE
	status = gpufit
		(
			n_fits,
			n_points,
			data.data(),
			// weights.data(),
			0,
			PATLAK,
			initial_parameters.data(),
			tolerance,
			max_n_iterations,
			parameters_to_fit.data(),
			MLE,
			n_points * sizeof(REAL),
			reinterpret_cast< char * >(user_info.data()),
			output_parameters.data(),
			output_states.data(),
			output_chi_squares.data(),
			output_n_iterations.data()
		);

	BOOST_CHECK(status == 0);
	// BOOST_CHECK(output_states == 0);
	// BOOST_CHECK(output_n_iterations <= max_n_iterations);
	// BOOST_CHECK(output_chi_squares < 1e-6);
	std::cout << "XD: " << output_parameters[0] << " " << true_parameters[0];
	BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-6);
	BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6);

}
