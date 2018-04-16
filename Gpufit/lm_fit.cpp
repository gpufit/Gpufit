#include "lm_fit.h"
#include <algorithm>
#include "../Cpufit/profile.h"

LMFit::LMFit
(
    REAL const * const data,
    REAL const * const weights,
    Info & info,
    REAL const * const initial_parameters,
    int const * const parameters_to_fit,
    char * const user_info,
    REAL * output_parameters,
    int * output_states,
    REAL * output_chi_squares,
    int * output_n_iterations
) :
    data_( data ),
    weights_( weights ),
    initial_parameters_( initial_parameters ),
    parameters_to_fit_( parameters_to_fit ),
    user_info_( user_info ),
    output_parameters_( output_parameters ),
    output_states_( output_states ),
    output_chi_squares_( output_chi_squares ),
    output_n_iterations_( output_n_iterations ),
    info_(info),
    chunk_size_(0),
    ichunk_(0),
    n_fits_left_(info.n_fits_),
    parameters_to_fit_indices_(0)
{}

LMFit::~LMFit()
{}

void LMFit::set_parameters_to_fit_indices()
{
    int const n_parameters_to_fit = info_.n_parameters_;
    for (int i = 0; i < n_parameters_to_fit; i++)
    {
        if (parameters_to_fit_[i])
        {
            parameters_to_fit_indices_.push_back(i);
        }
    }
}

void LMFit::get_results(GPUData const & gpu_data, int const n_fits)
{
    if (info_.data_location_ == HOST)
    {
        output_parameters_
            = gpu_data.parameters_.copy(n_fits*info_.n_parameters_, output_parameters_);
        output_states_
            = gpu_data.states_.copy(n_fits, output_states_);
        output_chi_squares_
            = gpu_data.chi_squares_.copy(n_fits, output_chi_squares_);
        output_n_iterations_
            = gpu_data.n_iterations_.copy(n_fits, output_n_iterations_);
    }
}

void LMFit::run(REAL const tolerance)
{
	std::chrono::high_resolution_clock::time_point t1, t2, t3, t4, t5, t6;

	t1 = std::chrono::high_resolution_clock::now();

    set_parameters_to_fit_indices();

	t2 = std::chrono::high_resolution_clock::now();

    GPUData gpu_data(info_);

    t3 = std::chrono::high_resolution_clock::now();

    gpu_data.init_user_info(user_info_);

    t4 = std::chrono::high_resolution_clock::now();

    profiler.initialize_LM += t2 - t1;
    profiler.allocate_GPU_memory += t3 - t2;
    profiler.copy_data_to_GPU += t4 - t3;

    // loop over data chunks
    while (n_fits_left_ > 0)
    {
        chunk_size_ = int((std::min)(n_fits_left_, info_.max_chunk_size_));

        t1 = std::chrono::high_resolution_clock::now();

        info_.set_fits_per_block(chunk_size_);

        t2 = std::chrono::high_resolution_clock::now();
        gpu_data.init(
            chunk_size_,
            ichunk_,
            data_,
            weights_,
            initial_parameters_,
            parameters_to_fit_indices_,
            output_states_,
            output_chi_squares_,
            output_n_iterations_);

        t3 = std::chrono::high_resolution_clock::now();

        LMFitCUDA lmfit_cuda(
            tolerance,
            info_,
            gpu_data,
            chunk_size_);

        t4 = std::chrono::high_resolution_clock::now();

        lmfit_cuda.run();

        t5 = std::chrono::high_resolution_clock::now();

        get_results(gpu_data, chunk_size_);

        t6 = std::chrono::high_resolution_clock::now();

        n_fits_left_ -= chunk_size_;
        ichunk_++;

        profiler.initialize_LM += t2 - t1 + t4 - t3;
        profiler.copy_data_to_GPU += t3 - t2;
        profiler.read_results_from_GPU += t6 - t5;
    }
}
