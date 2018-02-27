#ifndef CPUFIT_GAUSS_FIT_H_INCLUDED
#define CPUFIT_GAUSS_FIT_H_INCLUDED

#include "info.h"

class LMFitCPP;

class LMFit
{
public:
    LMFit(
        double const * data,
        double const * weights,
        Info const& info,
        double const * initial_parameters,
        int const * parameters_to_fit,
        char * user_info,
        double * output_parameters,
        int * output_states,
        double * output_chi_squares,
        int * output_n_iterations,
        double * lambda_info);

    virtual ~LMFit();

    void run(double const tolerance);
        
private:
    double const * const data_;
    double const * const weights_;
    double const * const initial_parameters_;
    int const * const parameters_to_fit_;
    char * const user_info_;

    double * output_parameters_;
    int * output_states_;
    double * output_chi_squares_;
    int * output_n_iterations_;
    double * lambda_info_;

    Info const & info_;
};

class LMFitCPP
{
public:
    LMFitCPP(
        double const tolerance,
        std::size_t const fit_index,
        double const * data,
        double const * weight,
        Info const & info,
        double const * initial_parameters,
        int const * parameters_to_fit,
        char * user_info,
        double * output_parameters,
        int * output_states,
        double * output_chi_squares,
        int * output_n_iterations,
        double * lambda_info);

    virtual ~LMFitCPP()
    {};

    void run();

private:
	void calc_model();
    void calc_coefficients();

    void calc_curve_values(std::vector<double>& curve, std::vector<double>& derivatives);

    void calc_values_gauss2d(std::vector<double>& gaussian);
    void calc_derivatives_gauss2d(std::vector<double> & derivatives);

    void calc_values_gauss2delliptic(std::vector<double>& gaussian);
    void calc_derivatives_gauss2delliptic(std::vector<double> & derivatives);

    void calc_values_gauss2drotated(std::vector<double>& gaussian);
    void calc_derivatives_gauss2drotated(std::vector<double> & derivatives);

    void calc_values_gauss1d(std::vector<double>& gaussian);
    void calc_derivatives_gauss1d(std::vector<double> & derivatives);

    void calc_values_cauchy2delliptic(std::vector<double>& cauchy);
    void calc_derivatives_cauchy2delliptic(std::vector<double> & derivatives);

    void calc_values_linear1d(std::vector<double>& line);
    void calc_derivatives_linear1d(std::vector<double> & derivatives);

    void calc_values_fletcher_powell_helix(std::vector<double>& values);
    void calc_derivatives_fletcher_powell_helix(std::vector<double> & derivatives);

    void calc_values_brown_dennis(std::vector<double>& values);
    void calc_derivatives_brown_dennis(std::vector<double> & derivatives);

    void calc_values_ramsey_var_p(std::vector<double>& values);
    void calc_derivatives_ramsey_var_p(std::vector<double> & derivatives);

    void calculate_hessian(std::vector<double> const & derivatives,
        std::vector<double> const & curve);

    void calc_gradient(std::vector<double> const & derivatives,
        std::vector<double> const & curve);

    void calc_chi_square(
        std::vector<double> const & curve);

    template< class T >
    void decompose_hessian_LUP(std::vector<T> & decomposed_hessian, std::vector<T> const & hessian);

    void modify_step_width();
    void update_parameters();

    bool check_for_convergence();
    void evaluate_iteration(int const iteration);
    void prepare_next_iteration();

    void calc_approximation_quality();
    void initialize_step_bound();
    void initialize_lambda_bounds();
    void update_step_bound();
    void calc_phi();
    void update_lambda();

public:

private:

    std::size_t const fit_index_;
    double const * const data_;
    double const * const weight_;
    double const * const initial_parameters_;
    int const * const parameters_to_fit_;

    bool converged_;
    double * parameters_;
    int * state_;
    double * chi_square_;
    int * n_iterations_;

    double * lambda_info_;
    double * output_lambda_;
    double * output_lower_bound_;
    double * output_upper_bound_;
    double * output_step_bound_;
    double * output_predicted_reduction_;
    double * output_actual_reduction_;
    double * output_directive_derivative_;
    double * output_phi_;
    double * output_chi_;
    double * output_iteration_;

    std::vector<double> prev_parameters_;
    Info const & info_;

    double lambda_;
    double lambda_lower_bound_;
    double lambda_upper_bound_;
    double step_bound_;
    double actual_reduction_;
    double predicted_reduction_;
    double directive_derivative_;
    double approximation_ratio_;
    double phi_;
    double phi_derivative_;

    std::vector<double> curve_;
    std::vector<double> derivatives_;
    std::vector<double> temp_derivatives_;
    std::vector<double> hessian_;
    std::vector<double> decomposed_hessian_;
    std::vector<double> inverted_hessian_;
    std::vector<int> pivot_array_;
    std::vector<double> modified_hessian_;
    std::vector<double> gradient_;
    std::vector<double> delta_;
    std::vector<double> scaling_vector_;
    double prev_chi_square_;
    double const tolerance_;

    char * const user_info_;
};

#endif