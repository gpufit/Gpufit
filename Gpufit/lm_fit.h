#ifndef GPUFIT_LM_FIT_H_INCLUDED
#define GPUFIT_LM_FIT_H_INCLUDED

#include "definitions.h"
#include "info.h"
#include "gpu_data.cuh"

class LMFitCUDA;

class LMFit
{
public:
    LMFit
    (
        float const * data,
        float const * weights,
        Info & info,
        float const * initial_parameters,
        int const * parameters_to_fit,
        char * user_info,
        float * output_parameters,
        int * output_states,
        float * output_chi_squares,
        int * output_n_iterations
    ) ;

    virtual ~LMFit();

    void run(float const tolerance);

private:
    void set_parameters_to_fit_indices();
    void get_results(GPUData const & gpu_data, int const n_fits);

    float const * const data_ ;
    float const * const weights_ ;
    float const * const initial_parameters_ ;
    int const * const parameters_to_fit_;
    char const * const user_info_;

    float * output_parameters_ ;
    int * output_states_ ;
    float * output_chi_squares_ ;
    int * output_n_iterations_ ;

    int ichunk_;
    int chunk_size_;
    std::size_t n_fits_left_;

    Info & info_;

    std::vector<int> parameters_to_fit_indices_;
};

class LMFitCUDA
{
public:
    LMFitCUDA(
        float const tolerance,
        Info const & info,
        GPUData & gpu_data,
        int const n_fits);

    virtual ~LMFitCUDA();

    void run();

private:
	void calc_curve_values();
    void calc_chi_squares();
    void calc_gradients();
    void calc_hessians();
    void calc_scaling_vectors();
    void evaluate_iteration(int const iteration);

    void scale_hessians();
    void decompose_hessians_LUP(Device_Array<float> const & hessians);
    void solve_equation_system();
    void update_parameters();
    void invert_hessians();

    void initialize_step_bounds();
    void adapt_step_bounds();
    void update_step_bounds();
    void calc_phi();
    void adapt_phi_derivatives();
    void check_phi();
    void check_abs_phi();
    void initialize_lambda_bounds();
    void update_lambdas();
    void check_all_lambdas();
    void calc_approximation_quality();

public:

private:
    Info const & info_;
    GPUData & gpu_data_;
    int const n_fits_;

    bool all_finished_;
    bool all_lambdas_accepted_;

    float tolerance_;
};

#endif
