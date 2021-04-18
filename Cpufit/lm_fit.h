#ifndef CPUFIT_GAUSS_FIT_H_INCLUDED
#define CPUFIT_GAUSS_FIT_H_INCLUDED

#ifdef _WIN64
#define SOLVE_EQUATION_SYSTEM() solve_equation_system_lup()
#else
#define SOLVE_EQUATION_SYSTEM() solve_equation_system_gj()
#endif // _WIN64

#include "info.h"

class LMFitCPP;

class LMFit
{
public:
    LMFit(
        REAL const * data,
        REAL const * weights,
        Info const& info,
        REAL const * initial_parameters,
        int const * parameters_to_fit,
        char * user_info,
        REAL * output_parameters,
        int * output_states,
        REAL * output_chi_squares,
        int * output_n_iterations);

    virtual ~LMFit();

    void run(REAL const tolerance);
        
private:
    REAL const * const data_;
    REAL const * const weights_;
    REAL const * const initial_parameters_;
    int const * const parameters_to_fit_;
    char * const user_info_;

    REAL * output_parameters_;
    int * output_states_;
    REAL * output_chi_squares_;
    int * output_n_iterations_;

    Info const & info_;
};

class LMFitCPP
{
public:
    LMFitCPP(
        REAL const tolerance,
        std::size_t const fit_index,
        REAL const * data,
        REAL const * weight,
        Info const & info,
        REAL const * initial_parameters,
        int const * parameters_to_fit,
        char * user_info,
        REAL * output_parameters,
        int * output_states,
        REAL * output_chi_squares,
        int * output_n_iterations);

    virtual ~LMFitCPP()
    {};

    void run();

private:
	void calc_model();
    void calc_coefficients();

    void calc_curve_values(std::vector<REAL>& curve, std::vector<REAL>& derivatives);

    void calc_values_gauss2d(std::vector<REAL>& gaussian);
    void calc_derivatives_gauss2d(std::vector<REAL> & derivatives);

    void calc_values_gauss2delliptic(std::vector<REAL>& gaussian);
    void calc_derivatives_gauss2delliptic(std::vector<REAL> & derivatives);

    void calc_values_gauss2drotated(std::vector<REAL>& gaussian);
    void calc_derivatives_gauss2drotated(std::vector<REAL> & derivatives);

    void calc_values_gauss1d(std::vector<REAL>& gaussian);
    void calc_derivatives_gauss1d(std::vector<REAL> & derivatives);

    void calc_values_cauchy2delliptic(std::vector<REAL>& cauchy);
    void calc_derivatives_cauchy2delliptic(std::vector<REAL> & derivatives);

    void calc_values_linear1d(std::vector<REAL>& line);
    void calc_derivatives_linear1d(std::vector<REAL> & derivatives);

    void calc_values_fletcher_powell_helix(std::vector<REAL>& values);
    void calc_derivatives_fletcher_powell_helix(std::vector<REAL> & derivatives);

    void calc_values_brown_dennis(std::vector<REAL>& values);
    void calc_derivatives_brown_dennis(std::vector<REAL> & derivatives);
    
    void calc_values_spline1d(std::vector<REAL>& values);
    void calc_derivatives_spline1d(std::vector<REAL> & derivatives);

    void calc_values_spline2d(std::vector<REAL>& values);
    void calc_derivatives_spline2d(std::vector<REAL> & derivatives);

    void calc_values_spline3d(std::vector<REAL>& values);
    void calc_derivatives_spline3d(std::vector<REAL> & derivatives);

    void calc_values_spline3d_multichannel(std::vector<REAL>& values);
    void calc_derivatives_spline3d_multichannel(std::vector<REAL> & derivatives);

    void calculate_hessian(std::vector<REAL> const & derivatives,
        std::vector<REAL> const & curve);

    void calc_gradient(std::vector<REAL> const & derivatives,
        std::vector<REAL> const & curve);

    void calc_chi_square(
        std::vector<REAL> const & curve);

    void decompose_hessian_LUP(std::vector<REAL> const & hessian);

    void modify_step_width();
    void solve_equation_system_gj();
    void solve_equation_system_lup();
    void update_parameters();

    bool check_for_convergence();
    void evaluate_iteration(int const iteration);
    void prepare_next_iteration();

public:

private:

    std::size_t const fit_index_;
    REAL const * const data_;
    REAL const * const weight_;
    REAL const * const initial_parameters_;
    int const * const parameters_to_fit_;

    bool converged_;
    REAL * parameters_;
    int * state_;
    REAL * chi_square_;
    int * n_iterations_;

    std::vector<REAL> prev_parameters_;
    Info const & info_;

    REAL lambda_;
    std::vector<REAL> curve_;
    std::vector<REAL> derivatives_;
    std::vector<REAL> hessian_;
    std::vector<REAL> decomposed_hessian_;
    std::vector<int> pivot_array_;
    std::vector<REAL> modified_hessian_;
    std::vector<REAL> gradient_;
    std::vector<REAL> delta_;
    std::vector<REAL> scaling_vector_;
    REAL prev_chi_square_;
    REAL const tolerance_;

    char * const user_info_;
};

#endif