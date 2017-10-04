#include "gpu_data.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

GPUData::GPUData(Info const & info) :
    chunk_size_(0),
    info_(info),

    data_( info_.max_chunk_size_*info_.n_points_ ),
    weights_( info_.use_weights_ ? info_.n_points_ * info_.max_chunk_size_ : 0 ),
    parameters_( info_.max_chunk_size_*info_.n_parameters_ ),
    prev_parameters_( info_.max_chunk_size_*info_.n_parameters_ ),
    parameters_to_fit_indices_( info_.n_parameters_to_fit_ ),
    user_info_( info_.user_info_size_ ),

    chi_squares_( info_.max_chunk_size_ * info_.n_blocks_per_fit_),
    prev_chi_squares_( info_.max_chunk_size_ ),
    gradients_( info_.max_chunk_size_ * info_.n_parameters_to_fit_ ),
    hessians_( info_.max_chunk_size_ * info_.n_parameters_to_fit_ * info_.n_parameters_to_fit_ ),
    deltas_(info_.max_chunk_size_ * info_.n_parameters_to_fit_),

    values_( info_.max_chunk_size_ * info_.n_points_ ),
    derivatives_( info_.max_chunk_size_ * info_.n_points_ * info_.n_parameters_ ),

    lambdas_( info_.max_chunk_size_ ),
    states_( info_.max_chunk_size_ ),
    finished_( info_.max_chunk_size_ ),
    iteration_failed_(info_.max_chunk_size_),
    all_finished_( 1 ),
    n_iterations_( info_.max_chunk_size_ )
{

}

void GPUData::reset(int const chunk_size)
{
    chunk_size_ = chunk_size;

    set(data_, 0.f, chunk_size_ * info_.n_points_);
    if (info_.use_weights_)
        set(weights_, 0.f, chunk_size_ * info_.n_points_);
    set(parameters_, 0.f, chunk_size_ * info_.n_parameters_);
    set(prev_parameters_, 0.f, chunk_size_ * info_.n_parameters_);
    set(parameters_to_fit_indices_, 0, info_.n_parameters_to_fit_);

    set(chi_squares_, 0.f, chunk_size_* info_.n_blocks_per_fit_);
    set(prev_chi_squares_, 0.f, chunk_size_);
    set(gradients_, 0.f, chunk_size_ * info_.n_parameters_to_fit_);
    set(hessians_, 0.f, chunk_size_ * info_.n_parameters_to_fit_ * info_.n_parameters_to_fit_);
    set(deltas_, 0.f, chunk_size_ * info_.n_parameters_to_fit_);

    set(values_, 0.f, chunk_size_*info_.n_points_);
    set(derivatives_, 0.f, chunk_size_ * info_.n_points_ * info_.n_parameters_);

    set(lambdas_, 0.f, chunk_size_);
    set(states_, 0, chunk_size_);
    set(finished_, 0, chunk_size_);
    set(iteration_failed_, 0, chunk_size_);
    set(all_finished_, 0, 1);
    set(n_iterations_, 0, chunk_size_);
}

void GPUData::init
(
    int const chunk_index,
    float const * const data,
    float const * const weights,
    float const * const initial_parameters,
    std::vector<int> const & parameters_to_fit_indices)
{
    chunk_index_ = chunk_index;
    write(
        data_,
        &data[chunk_index_*info_.max_chunk_size_*info_.n_points_],
        chunk_size_*info_.n_points_);
    if (info_.use_weights_)
        write(weights_, &weights[chunk_index_*info_.max_chunk_size_*info_.n_points_],
                chunk_size_*info_.n_points_);
    write(
        parameters_,
        &initial_parameters[chunk_index_*info_.max_chunk_size_*info_.n_parameters_],
        chunk_size_ * info_.n_parameters_);
    write(parameters_to_fit_indices_, parameters_to_fit_indices);

    set(lambdas_, 0.001f, chunk_size_);
}

void GPUData::init_user_info(char const * const user_info)
{
    if (info_.user_info_size_ > 0)
        write(user_info_, user_info, info_.user_info_size_);
}

void GPUData::read(bool * dst, int const * src)
{
    int int_dst = 0;
    CUDA_CHECK_STATUS(cudaMemcpy(&int_dst, src, sizeof(int), cudaMemcpyDeviceToHost));
    * dst = (int_dst == 1) ? true : false;
}

void GPUData::write(float* dst, float const * src, int const count)
{
    CUDA_CHECK_STATUS(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice));
}

void GPUData::write(int* dst, std::vector<int> const & src)
{
    std::size_t const size = src.size() * sizeof(int);
    CUDA_CHECK_STATUS(cudaMemcpy(dst, src.data(), size, cudaMemcpyHostToDevice));
}

void GPUData::write(char* dst, char const * src, std::size_t const count)
{
    CUDA_CHECK_STATUS(cudaMemcpy(dst, src, count * sizeof(char), cudaMemcpyHostToDevice));
}

void GPUData::copy(float * dst, float const * src, std::size_t const count)
{
    CUDA_CHECK_STATUS(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice));
}

__global__ void set_kernel(int* dst, int const value, int const count)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    dst[index] = value;
}

void GPUData::set(int* arr, int const value, int const count)
{
    int const tx = 256;
	int const bx = (count / tx) + 1;

    dim3  threads(tx, 1, 1);
    dim3  blocks(bx, 1, 1);

    set_kernel<<< blocks, threads >>>(arr, value, count);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

void GPUData::set(int* arr, int const value)
{
    int const tx = 1;
    int const bx = 1;

    dim3  threads(tx, 1, 1);
    dim3  blocks(bx, 1, 1);

    set_kernel<<< blocks, threads >>>(arr, value, 1);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

__global__ void set_kernel(float* dst, float const value, std::size_t const count)
{
	std::size_t const index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    dst[index] = value;
}

void GPUData::set(float* arr, float const value, int const count)
{
    int const tx = 256;
	int const bx = (count / tx) + 1;

    dim3  threads(tx, 1, 1);
    dim3  blocks(bx, 1, 1);
    set_kernel<<< blocks, threads >>>(arr, value, count);
    CUDA_CHECK_STATUS(cudaGetLastError());
}
