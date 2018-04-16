#include "lm_fit.h"
#include "../Cpufit/profile.h"

LMFitCUDA::LMFitCUDA(
    REAL const tolerance,
    Info const & info,
    GPUData & gpu_data,
    int const n_fits
    ) :
    info_(info),
    gpu_data_(gpu_data),
    n_fits_(n_fits),
    all_finished_(false),
    tolerance_(tolerance)
{
}

LMFitCUDA::~LMFitCUDA()
{
}

void LMFitCUDA::run()
{
	std::chrono::high_resolution_clock::time_point t1, t2, t3, t4, t5, t6, t7;

    // initialize the chi-square values
	t1 = std::chrono::high_resolution_clock::now();

	calc_curve_values();

    t2 = std::chrono::high_resolution_clock::now();

    calc_chi_squares();

    t3 = std::chrono::high_resolution_clock::now();

    calc_gradients();

    t4 = std::chrono::high_resolution_clock::now();

    calc_hessians();

	t5 = std::chrono::high_resolution_clock::now();

    profiler.compute_model += t2 - t1;
    profiler.compute_chisquare += t3 - t2;
    profiler.compute_gradient += t4 - t3;
    profiler.compute_hessian += t5 - t4;


    gpu_data_.copy(
        gpu_data_.prev_chi_squares_,
        gpu_data_.chi_squares_,
        n_fits_);

    // loop over the fit iterations
    for (int iteration = 0; !all_finished_; iteration++)
    {
        // modify step width
        // LUP decomposition
        // update fitting parameters
		t1 = std::chrono::high_resolution_clock::now();
        scale_hessians();
        SOLVE_EQUATION_SYSTEMS();
        update_states();
        update_parameters();

		t2 = std::chrono::high_resolution_clock::now();
		
        // calculate fitting curve values and its derivatives
        // calculate chi-squares, gradients and hessians
		calc_curve_values();

        t3 = std::chrono::high_resolution_clock::now();

        calc_chi_squares();

        t4 = std::chrono::high_resolution_clock::now();

        calc_gradients();

        t5 = std::chrono::high_resolution_clock::now();

        calc_hessians();

		t6 = std::chrono::high_resolution_clock::now();

        // check which fits have converged
        // flag finished fits
        // check whether all fits finished
        // save the number of needed iterations by each fitting process
        // check whether chi-squares are increasing or decreasing
        // update chi-squares, curve parameters and lambdas
        evaluate_iteration(iteration);

		t7 = std::chrono::high_resolution_clock::now();

        profiler.gauss_jordan += t2 - t1;
        profiler.compute_model += t3 - t2;
        profiler.compute_chisquare += t4 - t3;
        profiler.compute_gradient += t5 - t4;
        profiler.compute_hessian += t6 - t5;
        profiler.evaluate_iteration += t7 - t6;
    }
}