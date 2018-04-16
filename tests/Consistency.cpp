#define BOOST_TEST_MODULE Gpufit

#include "Cpufit/cpufit.h"
#include "Gpufit/gpufit.h"
#include "tests/utils.h"

#include <boost/test/included/unit_test.hpp>

#include <vector>

void generate_input_linear_fit_1d(FitInput & i)
{
	// number fits, points, parameters
	i.n_fits = 1;
	i.n_points = 2;
	i.n_parameters = 2; // LINEAR_1D has two parameters

	// data and weights
	i.data = { 0, 1 };
	i.weights_ = { 1, 1 };

	// model id and estimator id
	i.model_id = LINEAR_1D;
	i.estimator_id = LSE;

	// initial parameters and parameters to fit
	i.initial_parameters = { 0, 0 };
	i.parameters_to_fit = { 1, 1 };

	// tolerance and max_n_iterations
	i.tolerance = 0.001f;
	i.max_n_iterations = 10;

	// user info
	i.user_info_ = { 0., 1. };
}

void generate_input_gauss_fit_1d(FitInput & i)
{
	// number fits, points, parameters
	i.n_fits = 1;
	i.n_points = 5;
	i.n_parameters = 4; // GAUSS_1D has four parameters

	// data and weights
	clean_resize(i.data, i.n_fits * i.n_points);
	std::vector< REAL > const true_parameters{ { 4., 2., 0.5, 1. } };
	generate_gauss_1d(i.data, true_parameters);
	i.weights_.clear(); // no weights

	// model id and estimator id
	i.model_id = GAUSS_1D;
	i.estimator_id = LSE;

	// initial parameters and parameters to fit
	i.initial_parameters = { 2., 1.5, 0.3f, 0. };
	i.parameters_to_fit = { 1, 1, 1, 1 };

	// tolerance and max_n_iterations
	i.tolerance = 0.001f;
	i.max_n_iterations = 10;

	// user info
	i.user_info_.clear(); // no user info
}

void generate_input_gauss_fit_2d(FitInput & i)
{
	// number fits, points, parameters
	i.n_fits = 1;
	i.n_points = 25;
	i.n_parameters = 5; // GAUSS_2D has five parameters

	// data and weights
	clean_resize(i.data, i.n_fits * i.n_points);
	std::vector< REAL > const true_parameters{ { 4., 1.8f, 2.2f, 0.5, 1. } };
	generate_gauss_2d(i.data, true_parameters);
	i.weights_.clear(); // no weights

	// model id and estimator id
	i.model_id = GAUSS_2D;
	i.estimator_id = LSE;

	// initial parameters and parameters to fit
	i.initial_parameters = { 2., 1.8f, 2.2f, 0.4f, 0. };
	i.parameters_to_fit = { 1, 1, 1, 1, 1 };

	// tolerance and max_n_iterations
	i.tolerance = 0.0001f;
	i.max_n_iterations = 20;

	// user info
	i.user_info_.clear(); // no user info
}

void generate_input_gauss_fit_2d_elliptic(FitInput & i)
{
    // number fits, points, parameters
    i.n_fits = 1;
    std::size_t const size_x = 5;
    i.n_points = size_x * size_x;
    i.n_parameters = 6; // GAUSS_2D_ELLIPTIC has five parameters

    // data and weights
    clean_resize(i.data, i.n_fits * i.n_points);

    REAL const center_x = (static_cast<REAL>(size_x) - 1) / 2;
    std::vector< REAL > const true_parameters{ { 4, center_x, center_x, 0.4f, 0.6f, 1} };
    generate_gauss_2d_elliptic(i.data, true_parameters);
    i.weights_.clear(); // no weights

    // model id and estimator id
    i.model_id = GAUSS_2D_ELLIPTIC;
    i.estimator_id = LSE;

    // initial parameters and parameters to fit
    i.initial_parameters = { 2, 1.8f, 2.2f, .5f, .5f, 0 };
    i.parameters_to_fit = { 1, 1, 1, 1, 1 };

    // tolerance and max_n_iterations
    i.tolerance = 0.001f;
    i.max_n_iterations = 10;

    // user info
    i.user_info_.clear(); // no user info
}

void perform_cpufit_gpufit_and_check(void (*func)(FitInput &))
{
	// generate the data
	FitInput i;
	func(i);

	// sanity checks (we don't want to introduce faulty data)
	BOOST_CHECK(i.sanity_check());
	
	// reset output variables
	FitOutput gpu, cpu;
	clean_resize(gpu.parameters, i.n_fits * i.n_parameters);
	clean_resize(gpu.states, i.n_fits);
	clean_resize(gpu.chi_squares, i.n_fits);
	clean_resize(gpu.n_iterations, i.n_fits);

	clean_resize(cpu.parameters, i.n_fits * i.n_parameters);
	clean_resize(cpu.states, i.n_fits);
	clean_resize(cpu.chi_squares, i.n_fits);
	clean_resize(cpu.n_iterations, i.n_fits);


	// call to cpufit, store output
	int const cpu_status
		= cpufit
		(
			i.n_fits,
			i.n_points,
			i.data.data(),
			i.weights(),
			i.model_id,
			i.initial_parameters.data(),
			i.tolerance,
			i.max_n_iterations,
			i.parameters_to_fit.data(),
			i.estimator_id,
			i.user_info_size(),
			i.user_info(),
			cpu.parameters.data(),
			cpu.states.data(),
			cpu.chi_squares.data(),
			cpu.n_iterations.data()
		);

	BOOST_CHECK(cpu_status == 0);

	// call to gpufit, store output
	int const gpu_status
		= gpufit
		(
			i.n_fits,
			i.n_points,
			i.data.data(),
			i.weights(),
			i.model_id,
			i.initial_parameters.data(),
			i.tolerance,
			i.max_n_iterations,
			i.parameters_to_fit.data(),
			i.estimator_id,
			i.user_info_size(),
			i.user_info(),
			gpu.parameters.data(),
			gpu.states.data(),
			gpu.chi_squares.data(),
			gpu.n_iterations.data()
		);

	BOOST_CHECK(gpu_status == 0);

	// check both output for equality
	BOOST_CHECK(cpu.states == gpu.states);
	BOOST_CHECK(cpu.n_iterations == gpu.n_iterations);
	BOOST_CHECK(close_or_equal(cpu.parameters, gpu.parameters));
	BOOST_CHECK(close_or_equal(cpu.chi_squares, gpu.chi_squares));

}

BOOST_AUTO_TEST_CASE( Consistency )
{
	BOOST_TEST_MESSAGE( "linear_fit_1d" );
	perform_cpufit_gpufit_and_check(&generate_input_linear_fit_1d);

	BOOST_TEST_MESSAGE( "gauss_fit_1d" );
	perform_cpufit_gpufit_and_check(&generate_input_gauss_fit_1d);

	BOOST_TEST_MESSAGE( "gauss_fit_2d" );
	perform_cpufit_gpufit_and_check(&generate_input_gauss_fit_2d);

    BOOST_TEST_MESSAGE("gauss_fit_2d_elliptic");
    perform_cpufit_gpufit_and_check(&generate_input_gauss_fit_2d_elliptic);

}
