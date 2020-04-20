#ifndef GPUFIT_LM_FIT_H_INCLUDED
#define GPUFIT_LM_FIT_H_INCLUDED

#include "info.h"
#include "gpu_data.cuh"

#ifdef USE_CUBLAS
#include "cublas_v2.h"
#else
#include "cuda_gaussjordan.cuh"
#endif // USE_CUBLAS

class LMFitCUDA;

class LMFit
{
public:
    LMFit
    (
        REAL const * data,
        REAL const * weights,
        Info & info,
        REAL const * initial_parameters,
        int const * parameters_to_fit,
        REAL const * constraints,
        int const * constraint_types,
        char * user_info,
        REAL * output_parameters,
        int * output_states,
        REAL * output_chi_squares,
        int * output_n_iterations
    ) ;

    virtual ~LMFit();

    void run(REAL const tolerance);

private:
    void set_parameters_to_fit_indices();
    void get_results(GPUData const & gpu_data, int const n_fits);

    REAL const * const data_ ;
    REAL const * const weights_ ;
    REAL const * const initial_parameters_ ;
    int const * const parameters_to_fit_;
    REAL const * const constraints_;
    int const * const constraint_types_;
    char const * const user_info_;

    REAL * output_parameters_ ;
    int * output_states_ ;
    REAL * output_chi_squares_ ;
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
        REAL const tolerance,
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
    void evaluate_iteration(int const iteration);
    void scale_hessians();
    void solve_equation_systems_gj();
    void solve_equation_systems_lup();
    void update_states();
    void update_parameters();
    void project_parameters_to_box();

public:

private:
    Info const & info_;
    GPUData & gpu_data_;
    int const n_fits_;

    bool all_finished_;

    REAL tolerance_;
};

#endif
