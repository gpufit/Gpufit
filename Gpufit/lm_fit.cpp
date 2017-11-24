#include "lm_fit.h"
#include <algorithm>
#include <iostream>
#include "gpu_data.cuh"

LMFit::LMFit
(
    float const * const data,
    float const * const weights,
    Info & info,
    float const * const initial_parameters,
    int const * const parameters_to_fit,
    char * const user_info,
    float * output_parameters,
    int * output_states,
    float * output_chi_squares,
    int * output_n_iterations,
    float * output_data
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
    output_data_(output_data),
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


void LMFit::read_out(float * dst, Device_Array< float > const src, int const size)
{
    cudaMemcpy(dst, src, sizeof(float)*size , cudaMemcpyDeviceToHost);
}

void LMFit::read_out(std::vector<float> * dst, Device_Array< float > const src, float const size)
{
    cudaMemcpy(dst, src, sizeof(float)*size , cudaMemcpyDeviceToHost);
}

void LMFit::read_out(int * dst, Device_Array< int > const src, int const size)
{
    cudaMemcpy(dst, src, sizeof(int)*size , cudaMemcpyDeviceToHost);
}

void LMFit::read_out(std::vector<int> * dst, Device_Array< int > const src, int const size)
{
    cudaMemcpy(dst, src, sizeof(int)*size , cudaMemcpyDeviceToHost);
}


void LMFit::get_results(GPUData const & gpu_data, int const n_fits)
{
    /*
    output_parameters_
        = gpu_data.parameters_.copy( n_fits*info_.n_parameters_, output_parameters_ ) ;
    output_states_ = gpu_data.states_.copy( n_fits, output_states_ ) ;
    output_chi_squares_ = gpu_data.chi_squares_.copy( n_fits, output_chi_squares_ ) ;
    output_n_iterations_ = gpu_data.n_iterations_.copy( n_fits, output_n_iterations_ ) ;
    output_data_ =  gpu_data.values_.copy( n_fits*info_.n_points_, output_data_ ) ;
    */

    /*
    CUDA_CHECK_STATUS(cudaMemcpy(output_parameters_, gpu_data.parameters_, sizeof(float)*n_fits*info_.n_parameters_ , cudaMemcpyDeviceToHost));
    CUDA_CHECK_STATUS(cudaMemcpy(output_states_, gpu_data.states_, sizeof(int)*n_fits , cudaMemcpyDeviceToHost));
    CUDA_CHECK_STATUS(cudaMemcpy(output_chi_squares_, gpu_data.chi_squares_, sizeof(float)*n_fits , cudaMemcpyDeviceToHost));
    CUDA_CHECK_STATUS(cudaMemcpy(output_n_iterations_, gpu_data.n_iterations_, sizeof(int)*n_fits , cudaMemcpyDeviceToHost));
    CUDA_CHECK_STATUS(cudaMemcpy(output_data_, gpu_data.values_, sizeof(int)*n_fits*info_.n_points_ , cudaMemcpyDeviceToHost));
    */

    read_out(output_parameters_,gpu_data.parameters_,n_fits*info_.n_parameters_);
    read_out(output_states_,gpu_data.states_,n_fits);
    read_out(output_chi_squares_,gpu_data.chi_squares_,n_fits);
    read_out(output_n_iterations_,gpu_data.n_iterations_,n_fits);
    read_out(output_data_,gpu_data.values_,n_fits*info_.n_points_);

    // PRINT DEBUG
    std::cout << "LMFit::get_results --> PARAMS: ";
    for (int i = 0; i < info_.n_parameters_; i++)
    {
        std::cout << output_parameters_[i] << ' ';
    }
    std::cout << "\n";
    std::cout << "LMFit::get_results --> STATES: "<< output_states_[0] <<"\n";
    std::cout << "LMFit::get_results --> CHI_SQ: "<< output_chi_squares_[0] <<"\n";
    std::cout << "LMFit::get_results --> ITERAT: "<< output_n_iterations_[0] <<"\n";
    std::cout << "LMFit::get_results --> VALUES: ";
    for (int i = 0; i < info_.n_points_; i++)
    {
        std::cout << output_data_[i] << ' ';
    }
    std::cout << "\n";
}

void LMFit::run(float const tolerance)
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
            chunk_size_);

        lmfit_cuda.run();
        get_results(gpu_data, chunk_size_);

        for (size_t i = 0; i != info_.n_fits_; i++)
        {
            std::cout << "LMFit::run --> iterations:"<< output_n_iterations_[i]<<"\n";
        }

        std::cout << "\n";

        n_fits_left_ -= chunk_size_;
        ichunk_++;
    }
}

void LMFit::simul(float const tolerance)
{
    std::cout << "LMFit::simul" <<"\n";
    set_parameters_to_fit_indices();

    GPUData gpu_data(info_);
    gpu_data.init_user_info(user_info_);


    //float * data_out[info_.n_fits_][info_.n_points_];

    // loop over data chunks
    while (n_fits_left_ > 0)
    {
        std::cout << "loop..." <<"\n";
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
            chunk_size_);

        lmfit_cuda.simul();
        get_results(gpu_data, chunk_size_);

        n_fits_left_ -= chunk_size_;
        ichunk_++;
    }
}
