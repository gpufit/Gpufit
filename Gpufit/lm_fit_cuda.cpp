#include "lm_fit.h"

LMFitCUDA::LMFitCUDA(
    float const tolerance,
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
    // calculate model using initial parameters
	calc_curve_values();

    // initialize temp derivatives
    gpu_data_.copy(gpu_data_.temp_derivatives_, gpu_data_.derivatives_, n_fits_ * info_.n_parameters_ * info_.n_points_);

    // calculate initial chi-squares, gradients and hessians
    calc_chi_squares();
    calc_gradients();
    calc_hessians();

    // initialize scaling vectors
    calc_scaling_vectors();

    // initialize scaled hessians
    scale_hessians();

    // initialize previous chi-squares
    gpu_data_.copy(
        gpu_data_.prev_chi_squares_,
        gpu_data_.chi_squares_,
        n_fits_);

    // initialize step bounds
    initialize_step_bounds();

    // loop over the fit iterations
    for (int iteration = 0; !all_finished_; iteration++)
    {
        // decompose hessians by LUP decomposition
        decompose_hessians_LUP(gpu_data_.hessians_);

        // solve equation systems (hessian * delta = gradient) for delta
        solve_equation_system();

        // iinvert hessians
        invert_hessians();

        // update scaling vectors
        calc_scaling_vectors();

        // calculate phi and its derivative
        calc_phi();

        // multiply the derivative of phi by the factor step_bound / scaled_delta_norm
        adapt_phi_derivatives();

        // test for gauss-netwon step acceptance
        check_phi();

        // initialize lambda bounds and adapt lambda to them
        initialize_lambda_bounds();

        // scale hessian and solve the equation systems using scaled hessians
        scale_hessians();
        decompose_hessians_LUP(gpu_data_.scaled_hessians_);
        solve_equation_system();
        invert_hessians();

        // update phi and its derivatives
        calc_phi();

        // start loop for iterative estimation of suitable value for lambda
        for (int iter_lambda = 0; iter_lambda < 10; iter_lambda++)
        {
            // test for acceptance of the lambda values and get out of the loop
            // if lambda values for all fits are accepted
            check_abs_phi();
            check_all_lambdas();

            if (all_lambdas_accepted_)
                break;

            // update lambda values
            update_lambdas();

            // rescale hessians
            scale_hessians();

            // solve equation systems
            decompose_hessians_LUP(gpu_data_.scaled_hessians_);
            solve_equation_system();
            invert_hessians();

            // update phi and its derivatives
            calc_phi();
        }

        // at the first iteration adjust step bounds
        if (iteration == 0)
        {
            adapt_step_bounds();
        }

        // update parameters by delta
        update_parameters();

        // calculate the model using updated parameters
        calc_curve_values();

        // calculate chi-squares, gradients and hessians
        calc_chi_squares();
        calc_gradients();
        calc_hessians();

        // update scaling vectors
        calc_scaling_vectors();

        // calculate ratio actual_reduction / predicted reduction
        calc_approximation_quality();

        // update step bounds using an if-else sequence
        update_step_bounds();

        // check which fits have converged
        // flag finished fits
        // check whether all fits finished
        // save the number of needed iterations by each fitting process
        // check whether chi-squares are increasing or decreasing
        // in case of a successful iteration:
        //      update previous chi-squares previous fit parameters and temp
        //      derivatives
        // otherwise:
        //      revert values of chi-squares and fit parameters
        evaluate_iteration(iteration);

        // reset lambda_accepted and newton_step_accepted flags
        gpu_data_.set(gpu_data_.lambda_accepted_, 0, n_fits_);
        gpu_data_.set(gpu_data_.newton_step_accepted_, 1, n_fits_);
    }
}