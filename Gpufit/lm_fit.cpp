#include "lm_fit.h"
#include <algorithm>

LMFit::LMFit
(
    double const * const data,
    double const * const weights,
    Info & info,
    double const * const initial_parameters,
    int const * const parameters_to_fit,
    char * const user_info,
    double * output_parameters,
    int * output_states,
    double * output_chi_squares,
    int * output_n_iterations,
    double * lambda_info
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
    parameters_to_fit_indices_(0),
    lambda_info_(lambda_info)
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
    output_parameters_
        = gpu_data.parameters_.copy( n_fits*info_.n_parameters_, output_parameters_ ) ;
    output_states_ = gpu_data.states_.copy( n_fits, output_states_ ) ;
    output_chi_squares_ = gpu_data.chi_squares_.copy( n_fits, output_chi_squares_ ) ;
    output_n_iterations_ = gpu_data.n_iterations_.copy( n_fits, output_n_iterations_ ) ;
}

void LMFit::run(double const tolerance)
{
    set_parameters_to_fit_indices();

    GPUData gpu_data(info_);
    gpu_data.init_user_info(user_info_);

    // loop over data chunks
    while (n_fits_left_ > 0)
    {
        chunk_size_ = int((std::min)(n_fits_left_, info_.max_chunk_size_));

        info_.set_fits_per_block(chunk_size_);

        gpu_data.init(
            chunk_size_,
            ichunk_,
            data_,
            weights_,
            initial_parameters_,
            parameters_to_fit_indices_);

        LMFitCUDA lmfit_cuda(
            tolerance,
            info_,
            gpu_data,
            chunk_size_,
            lambda_info_ + ichunk_ * info_.max_chunk_size_ * 10 * 1000);

        lmfit_cuda.run();

        get_results(gpu_data, chunk_size_);

        n_fits_left_ -= chunk_size_;
        ichunk_++;
    }
}
