#include "lm_fit.h"

LMFitCUDA::LMFitCUDA(
    double const tolerance,
    Info const & info,
    GPUData & gpu_data,
    int const n_fits
    ) :
    info_(info),
    gpu_data_(gpu_data),
    n_fits_(n_fits),
    all_finished_(false),
    all_lambdas_accepted_(false),
    tolerance_(tolerance)
{
}

LMFitCUDA::~LMFitCUDA()
{
}

void LMFitCUDA::run()
{    
    // initialize the chi-square values
	calc_curve_values();

    gpu_data_.copy(gpu_data_.temp_derivatives_, gpu_data_.derivatives_, n_fits_ * info_.n_parameters_ * info_.n_points_);

    calc_chi_squares();
    calc_gradients();
    calc_hessians();
    calc_scaling_vectors();
    scale_hessians();

    gpu_data_.copy(
        gpu_data_.prev_chi_squares_,
        gpu_data_.chi_squares_,
        n_fits_);

    initialize_step_bounds();

    // loop over the fit iterations
    for (int iteration = 0; !all_finished_; iteration++)
    {
        gpu_data_.set(gpu_data_.lambda_accepted_, 0, n_fits_);
        gpu_data_.set(gpu_data_.newton_step_accepted_, 1, n_fits_);

        decompose_hessians_LUP(gpu_data_.hessians_);
        solve_equation_system();
        invert_hessians();

        calc_scaling_vectors();

        calc_phi();
        adapt_phi_derivatives();

        check_phi();
        initialize_lambda_bounds();

        calc_scaling_vectors();
        scale_hessians();

        decompose_hessians_LUP(gpu_data_.scaled_hessians_);
        solve_equation_system();
        invert_hessians();

        calc_phi();

        for (int iter_lambda = 0; iter_lambda < 10; iter_lambda++)
        {
            check_abs_phi();
            check_all_lambdas();

            std::vector<int> lambda_accepted(1);
            std::vector<double> phi(1);
            std::vector<double> lambda(1);
            gpu_data_.phis_.copy(1, phi.data());
            gpu_data_.lambdas_.copy(1, lambda.data());
            gpu_data_.lambda_accepted_.copy(1, lambda_accepted.data());

            if (all_lambdas_accepted_)
                break;

            update_lambdas();

            calc_scaling_vectors();
            scale_hessians();

            decompose_hessians_LUP(gpu_data_.scaled_hessians_);
            solve_equation_system();
            invert_hessians();

            calc_phi();
        }

        if (iteration == 0)
        {
            adapt_step_bounds();
        }

        update_parameters();

        // calculate fitting curve values and its derivatives
        // calculate chi-squares, gradients and hessians
        calc_curve_values();
        calc_chi_squares();
        calc_gradients();
        calc_hessians();

        calc_approximation_quality();
        update_step_bounds();

        // check which fits have converged
        // flag finished fits
        // check whether all fits finished
        // save the number of needed iterations by each fitting process
        // check whether chi-squares are increasing or decreasing
        // update chi-squares, curve parameters and lambdas
        evaluate_iteration(iteration);
    }
}