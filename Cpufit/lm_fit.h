#ifndef CPUFIT_GAUSS_FIT_H_INCLUDED
#define CPUFIT_GAUSS_FIT_H_INCLUDED

#include "info.h"

class LMFitCPP;

class LMFit
{
public:
    LMFit(
        float const * data,
        float const * weights,
        Info const& info,
        float const * initial_parameters,
        int const * parameters_to_fit,
        char * user_info,
        float * output_parameters,
        int * output_states,
        float * output_chi_squares,
        int * output_n_iterations);

    virtual ~LMFit();

    void run(float const tolerance);
        
private:
    float const * const data_;
    float const * const weights_;
    float const * const initial_parameters_;
    int const * const parameters_to_fit_;
    char * const user_info_;

    float * output_parameters_;
    int * output_states_;
    float * output_chi_squares_;
    int * output_n_iterations_;

    Info const & info_;
};

class LMFitCPP
{
public:
    LMFitCPP(
        float const tolerance,
        std::size_t const fit_index,
        float const * data,
        float const * weight,
        Info const & info,
        float const * initial_parameters,
        int const * parameters_to_fit,
        char * user_info,
        float * output_parameters,
        int * output_states,
        float * output_chi_squares,
        int * output_n_iterations);

    virtual ~LMFitCPP()
    {};

    void run();

private:
	void calc_curve_values();
    void calc_coefficients();

    void calc_curve_values(std::vector<float>& curve, std::vector<float>& derivatives);

    void calc_values_gauss2d(std::vector<float>& gaussian);
    void calc_derivatives_gauss2d(std::vector<float> & derivatives);

    void calc_values_gauss2delliptic(std::vector<float>& gaussian);
    void calc_derivatives_gauss2delliptic(std::vector<float> & derivatives);

    void calc_values_gauss2drotated(std::vector<float>& gaussian);
    void calc_derivatives_gauss2drotated(std::vector<float> & derivatives);

    void calc_values_gauss1d(std::vector<float>& gaussian);
    void calc_derivatives_gauss1d(std::vector<float> & derivatives);

    void calc_values_cauchy2delliptic(std::vector<float>& cauchy);
    void calc_derivatives_cauchy2delliptic(std::vector<float> & derivatives);

    void calc_values_linear1d(std::vector<float>& line);
    void calc_derivatives_linear1d(std::vector<float> & derivatives);

    void calculate_hessian(std::vector<float> const & derivatives,
        std::vector<float> const & curve);

    void calc_gradient(std::vector<float> const & derivatives,
        std::vector<float> const & curve);

    void calc_chi_square(
        std::vector<float> const & curve);

    void modify_step_width();
    void gauss_jordan();
    void update_parameters();

    bool check_for_convergence();
    void evaluate_iteration(int const iteration);
    void prepare_next_iteration();

public:

private:

    std::size_t const fit_index_;
    float const * const data_;
    float const * const weight_;
    float const * const initial_parameters_;
    int const * const parameters_to_fit_;

    bool converged_;
    float * parameters_;
    int * state_;
    float * chi_square_;
    int * n_iterations_;

    std::vector<float> prev_parameters_;
    Info const & info_;

    float lambda_;
    std::vector<float> curve_;
    std::vector<float> derivatives_;
    std::vector<float> hessian_;
    std::vector<float> modified_hessian_;
    std::vector<float> gradient_;
    std::vector<float> delta_;
    float prev_chi_square_;
    float const tolerance_;

    char * const user_info_;
};

#endif