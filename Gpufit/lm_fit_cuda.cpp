#include "lm_fit.h"

LMFitCUDA::LMFitCUDA(
    double const tolerance,
    Info const & info,
    GPUData & gpu_data,
    int const n_fits,
    double * lambda_info
    ) :
    info_(info),
    gpu_data_(gpu_data),
    n_fits_(n_fits),
    all_finished_(false),
    all_lambdas_accepted_(false),
    tolerance_(tolerance),
    lambda_info_(lambda_info),
    output_lambda_(lambda_info),
    output_lower_bound_(lambda_info + 1000),
    output_upper_bound_(lambda_info + 2000),
    output_step_bound_(lambda_info + 3000),
    output_predicted_reduction_(lambda_info + 4000),
    output_actual_reduction_(lambda_info + 5000),
    output_directive_derivative_(lambda_info + 6000),
    output_phi_(lambda_info + 7000),
    output_chi_(lambda_info + 8000),
    output_iteration_(lambda_info + 9000)
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

    /////////////////////////////////////////////////////////////////////////////////////**/
    /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
    /**/ *(output_lower_bound_++) = 0.;                                                /**/
    /**/ *(output_upper_bound_++) = 0.;                                                /**/
    /**/ *(output_step_bound_++) = 0.;                                                 /**/
    /**/ *(output_actual_reduction_++) = 0.;                                           /**/
    /**/ *(output_predicted_reduction_++) = 0.;                                        /**/
    /**/ *(output_directive_derivative_++) = 0.;                                       /**/
    /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
    /**/ *(output_phi_++) = 0.;                                                        /**/
    /**/ *(output_iteration_++) = -1;                                                   /**/
    /////////////////////////////////////////////////////////////////////////////////////**/

    gpu_data_.copy(
        gpu_data_.prev_chi_squares_,
        gpu_data_.chi_squares_,
        n_fits_);

    initialize_step_bounds();

    /////////////////////////////////////////////////////////////////////////////////////**/
    /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
    /**/ *(output_lower_bound_++) = 0.;                                                /**/
    /**/ *(output_upper_bound_++) = 0.;                                                /**/
    /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
    /**/ *(output_actual_reduction_++) = 0.;                                           /**/
    /**/ *(output_predicted_reduction_++) = 0.;                                        /**/
    /**/ *(output_directive_derivative_++) = 0.;                                       /**/
    /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
    /**/ *(output_phi_++) = 0.;                                                        /**/
    /**/ *(output_iteration_++) = 0;                                                    /**/
    /////////////////////////////////////////////////////////////////////////////////////**/

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

        /////////////////////////////////////////////////////////////////////////////////////**/
        /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
        /**/ *(output_lower_bound_++) = 0.;                                                /**/
        /**/ *(output_upper_bound_++) = 0.;                                                /**/
        /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
        /**/ *(output_actual_reduction_++) = *(output_actual_reduction_ - 1);               /**/
        /**/ *(output_predicted_reduction_++) = *(output_predicted_reduction_ - 1);         /**/
        /**/ *(output_directive_derivative_++) = *(output_directive_derivative_ - 1);       /**/
        /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
        /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
        /**/ *(output_iteration_++) = iteration;                                            /**/
        /////////////////////////////////////////////////////////////////////////////////////**/

        check_phi();
        initialize_lambda_bounds();

        /////////////////////////////////////////////////////////////////////////////////////**/
        /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
        /**/ gpu_data_.lambda_lower_bounds_.copy(1, output_lower_bound_++);                 /**/
        /**/ gpu_data_.lambda_upper_bounds_.copy(1, output_upper_bound_++);                 /**/
        /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
        /**/ *(output_actual_reduction_++) = *(output_actual_reduction_ - 1);               /**/
        /**/ *(output_predicted_reduction_++) = *(output_predicted_reduction_ - 1);         /**/
        /**/ *(output_directive_derivative_++) = *(output_directive_derivative_ - 1);       /**/
        /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
        /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
        /**/ *(output_iteration_++) = iteration;                                            /**/
        /////////////////////////////////////////////////////////////////////////////////////**/

        calc_scaling_vectors();
        scale_hessians();

        decompose_hessians_LUP(gpu_data_.scaled_hessians_);
        solve_equation_system();
        invert_hessians();

        calc_phi();

        /////////////////////////////////////////////////////////////////////////////////////**/
        /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
        /**/ gpu_data_.lambda_lower_bounds_.copy(1, output_lower_bound_++);                 /**/
        /**/ gpu_data_.lambda_upper_bounds_.copy(1, output_upper_bound_++);                 /**/
        /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
        /**/ *(output_actual_reduction_++) = *(output_actual_reduction_ - 1);               /**/
        /**/ *(output_predicted_reduction_++) = *(output_predicted_reduction_ - 1);         /**/
        /**/ *(output_directive_derivative_++) = *(output_directive_derivative_ - 1);       /**/
        /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
        /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
        /**/ *(output_iteration_++) = iteration;                                            /**/
        /////////////////////////////////////////////////////////////////////////////////////**/

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

            /////////////////////////////////////////////////////////////////////////////////////**/
            /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
            /**/ gpu_data_.lambda_lower_bounds_.copy(1, output_lower_bound_++);                 /**/
            /**/ gpu_data_.lambda_upper_bounds_.copy(1, output_upper_bound_++);                 /**/
            /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
            /**/ *(output_actual_reduction_++) = *(output_actual_reduction_ - 1);               /**/
            /**/ *(output_predicted_reduction_++) = *(output_predicted_reduction_ - 1);         /**/
            /**/ *(output_directive_derivative_++) = *(output_directive_derivative_ - 1);       /**/
            /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
            /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
            /**/ *(output_iteration_++) = iteration;                                            /**/
            /////////////////////////////////////////////////////////////////////////////////////**/

            calc_scaling_vectors();
            scale_hessians();

            decompose_hessians_LUP(gpu_data_.scaled_hessians_);
            solve_equation_system();
            invert_hessians();

            calc_phi();

            /////////////////////////////////////////////////////////////////////////////////////**/
            /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
            /**/ gpu_data_.lambda_lower_bounds_.copy(1, output_lower_bound_++);                 /**/
            /**/ gpu_data_.lambda_upper_bounds_.copy(1, output_upper_bound_++);                 /**/
            /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
            /**/ *(output_actual_reduction_++) = *(output_actual_reduction_ - 1);               /**/
            /**/ *(output_predicted_reduction_++) = *(output_predicted_reduction_ - 1);         /**/
            /**/ *(output_directive_derivative_++) = *(output_directive_derivative_ - 1);       /**/
            /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
            /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
            /**/ *(output_iteration_++) = iteration;                                            /**/
            /////////////////////////////////////////////////////////////////////////////////////**/
        }

        if (iteration == 0)
        {
            adapt_step_bounds();

            /////////////////////////////////////////////////////////////////////////////////////**/
            /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
            /**/ *(output_lower_bound_++) = *(output_lower_bound_ - 1);                         /**/
            /**/ *(output_upper_bound_++) = *(output_upper_bound_ - 1);                         /**/
            /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
            /**/ *(output_actual_reduction_++) = *(output_actual_reduction_ - 1);               /**/
            /**/ *(output_predicted_reduction_++) = *(output_predicted_reduction_ - 1);         /**/
            /**/ *(output_directive_derivative_++) = *(output_directive_derivative_ - 1);       /**/
            /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
            /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
            /**/ *(output_iteration_++) = iteration;                                            /**/
            /////////////////////////////////////////////////////////////////////////////////////**/
        }

        update_parameters();

        // calculate fitting curve values and its derivatives
        // calculate chi-squares, gradients and hessians
        calc_curve_values();
        calc_chi_squares();
        calc_gradients();
        calc_hessians();

        /////////////////////////////////////////////////////////////////////////////////////**/
        /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
        /**/ *(output_lower_bound_++) = *(output_lower_bound_ - 1);                         /**/
        /**/ *(output_upper_bound_++) = *(output_upper_bound_ - 1);                         /**/
        /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
        /**/ *(output_actual_reduction_++) = *(output_actual_reduction_ - 1);               /**/
        /**/ *(output_predicted_reduction_++) = *(output_predicted_reduction_ - 1);         /**/
        /**/ *(output_directive_derivative_++) = *(output_directive_derivative_ - 1);       /**/
        /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
        /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
        /**/ *(output_iteration_++) = iteration;                                            /**/
        /////////////////////////////////////////////////////////////////////////////////////**/

        calc_approximation_quality();
        update_step_bounds();

        /////////////////////////////////////////////////////////////////////////////////////**/
        /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
        /**/ *(output_lower_bound_++) = *(output_lower_bound_ - 1);                         /**/
        /**/ *(output_upper_bound_++) = *(output_upper_bound_ - 1);                         /**/
        /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
        /**/ gpu_data_.actual_reductions_.copy(1, output_actual_reduction_++);              /**/
        /**/ gpu_data_.predicted_reductions_.copy(1, output_predicted_reduction_++);        /**/
        /**/ gpu_data_.directive_derivatives_.copy(1, output_directive_derivative_++);      /**/
        /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
        /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
        /**/ *(output_iteration_++) = iteration;                                            /**/
        /////////////////////////////////////////////////////////////////////////////////////**/

        // check which fits have converged
        // flag finished fits
        // check whether all fits finished
        // save the number of needed iterations by each fitting process
        // check whether chi-squares are increasing or decreasing
        // update chi-squares, curve parameters and lambdas
        evaluate_iteration(iteration);

        /////////////////////////////////////////////////////////////////////////////////////**/
        /**/ gpu_data_.lambdas_.copy(1, output_lambda_++);                                  /**/
        /**/ *(output_lower_bound_++) = *(output_lower_bound_ - 1);                         /**/
        /**/ *(output_upper_bound_++) = *(output_upper_bound_ - 1);                         /**/
        /**/ gpu_data_.step_bounds_.copy(1, output_step_bound_++);                          /**/
        /**/ gpu_data_.actual_reductions_.copy(1, output_actual_reduction_++);              /**/
        /**/ gpu_data_.predicted_reductions_.copy(1, output_predicted_reduction_++);        /**/
        /**/ gpu_data_.directive_derivatives_.copy(1, output_directive_derivative_++);      /**/
        /**/ gpu_data_.chi_squares_.copy(1, output_chi_++);                                 /**/
        /**/ gpu_data_.phis_.copy(1, output_phi_++);                                        /**/
        /**/ *(output_iteration_++) = iteration;                                            /**/
        /////////////////////////////////////////////////////////////////////////////////////**/
    }
}