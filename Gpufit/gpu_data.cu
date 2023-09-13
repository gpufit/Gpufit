#include "gpu_data.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

GPUData::GPUData(Info const & info) :
    chunk_size_(0),
    info_(info),

    data_(
        (info_.data_location_ == HOST)
        ? info_.max_chunk_size_*info_.n_points_ : 0),
    weights_( 
        (info_.use_weights_ && info_.data_location_ == HOST)
        ? info_.n_points_ * info_.max_chunk_size_ : 0 ),
    parameters_(
        (info_.data_location_ == HOST)
        ? info_.max_chunk_size_*info_.n_parameters_ : 0 ),
    user_info_(
        (info_.data_location_ == HOST)
        ? info_.user_info_size_ : 0),

    prev_parameters_( info_.max_chunk_size_*info_.n_parameters_ ),
    parameters_to_fit_indices_( info_.n_parameters_to_fit_ ),

    constraints_(
        info_.use_constraints_
        ? info_.n_parameters_ * 2 : 0),
    constraint_types_(
        info_.use_constraints_
        ? info_.n_parameters_ : 0),

    chi_squares_(
        (info_.data_location_ == HOST)
        ? info_.max_chunk_size_ : 0),

    prev_chi_squares_( info_.max_chunk_size_ ),
    gradients_( info_.max_chunk_size_ * info_.n_parameters_to_fit_),
    hessians_( info_.max_chunk_size_ * info_.n_parameters_to_fit_ * info_.n_parameters_to_fit_ ),
    deltas_(info_.max_chunk_size_ * info_.n_parameters_to_fit_),
    scaling_vectors_(info_.max_chunk_size_ * info_.n_parameters_to_fit_),

    subtotals_(
        (info_.n_blocks_per_fit_ > 1)
        ? std::max(
            info_.max_chunk_size_ * info_.n_parameters_to_fit_ * info_.n_blocks_per_fit_,
            info_.max_chunk_size_ * info_.n_blocks_per_fit_)
        : 0),

    values_( info_.max_chunk_size_ * info_.n_points_ ),
    derivatives_( info_.max_chunk_size_ * info_.n_points_ * info_.n_parameters_ ),

    lambdas_( info_.max_chunk_size_ ),

    states_(
        (info_.data_location_ == HOST)
        ? info_.max_chunk_size_ : 0),
    
    finished_( info_.max_chunk_size_ ),
    iteration_failed_(info_.max_chunk_size_),
    all_finished_( 1 ),

    n_iterations_(
        (info_.data_location_ == HOST)
        ? info_.max_chunk_size_ : 0),
    
    solution_info_(info_.max_chunk_size_)

#ifdef USE_CUBLAS
    ,
    decomposed_hessians_(info_.max_chunk_size_ * info_.n_parameters_to_fit_ * info_.n_parameters_to_fit_),
    pointer_decomposed_hessians_(info_.max_chunk_size_),
    pointer_deltas_(info_.max_chunk_size_),
    pivot_vectors_(info_.max_chunk_size_ * info_.n_parameters_to_fit_)
#endif // USE_CUBLAS
{
#ifdef USE_CUBLAS
    cublasCreate(&cublas_handle_);
    point_to_data_sets();
#endif // USE_CUBLAS
}

GPUData::~GPUData()
{
#ifdef USE_CUBLAS
    cublasDestroy(cublas_handle_);
#endif // USE_CUBLAS
}

void GPUData::init
(
    int const chunk_size,
    int const chunk_index,
    REAL const * const data,
    REAL const * const weights,
    REAL const * const initial_parameters,
    std::vector<int> const & parameters_to_fit_indices,
    REAL const * const constraints,
    int const * const constraint_types,
    int * states,
    REAL * chi_squares,
    int * n_iterations)
{
    chunk_size_ = chunk_size;
    chunk_index_ = chunk_index;

    if (info_.data_location_ == HOST)
    {
        write(
            data_,
            data + chunk_index_*info_.max_chunk_size_*info_.n_points_,
            chunk_size_ * info_.n_points_);
        write(
            parameters_,
            initial_parameters + chunk_index_*info_.max_chunk_size_*info_.n_parameters_,
            chunk_size_ * info_.n_parameters_);
        if (info_.use_weights_)
            write(
                weights_,
                weights + chunk_index_*info_.max_chunk_size_*info_.n_points_,
                chunk_size_ * info_.n_points_);
    }
    else if (info_.data_location_ == DEVICE)
    {
        data_.assign(
            data + chunk_index_*info_.max_chunk_size_*info_.n_points_);
        parameters_.assign(
            initial_parameters + chunk_index_*info_.max_chunk_size_*info_.n_parameters_);
        if (info_.use_weights_)
            weights_.assign(
                weights + chunk_index_*info_.max_chunk_size_*info_.n_points_);
        states_.assign(
            states + chunk_index_ * info_.max_chunk_size_);
        chi_squares_.assign(
            chi_squares + chunk_index_ * info_.max_chunk_size_);
        n_iterations_.assign(
            n_iterations + chunk_index_ * info_.max_chunk_size_);
    }

    write(parameters_to_fit_indices_, parameters_to_fit_indices);

    if (info_.use_constraints_)
    {
        write(constraints_, constraints, 2 * info_.n_parameters_);
        write(constraint_types_, constraint_types, info_.n_parameters_);
    }

    set(prev_chi_squares_, 0., chunk_size_);
    set(finished_, 0, chunk_size_);
    set(scaling_vectors_, 0., chunk_size_ * info_.n_parameters_to_fit_);
    set(states_, 0, chunk_size_);
    set(lambdas_, 0.001f, chunk_size_);
    set(n_iterations_, 0, chunk_size_);
}

void GPUData::init_user_info(char const * const user_info)
{
    if (info_.user_info_size_ > 0)
    {
        if (info_.data_location_ == HOST)
        {
            write(user_info_, user_info, info_.user_info_size_);
        }
        else if (info_.data_location_ == DEVICE)
        {
            user_info_.assign(user_info);
        }
    }
}

void GPUData::read(bool * dst, int const * src)
{
    int int_dst = 0;
    CUDA_CHECK_STATUS(cudaMemcpy(&int_dst, src, sizeof(int), cudaMemcpyDeviceToHost));
    * dst = (int_dst == 1) ? true : false;
}

void GPUData::write(REAL* dst, REAL const * src, int const count)
{
    CUDA_CHECK_STATUS(cudaMemcpy(dst, src, count * sizeof(REAL), cudaMemcpyHostToDevice));
}

void GPUData::write(int * dst, int const * src, int const count)
{
    CUDA_CHECK_STATUS(cudaMemcpy(dst, src, count * sizeof(int), cudaMemcpyHostToDevice));
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

void GPUData::copy(REAL * dst, REAL const * src, std::size_t const count)
{
    CUDA_CHECK_STATUS(cudaMemcpy(dst, src, count * sizeof(REAL), cudaMemcpyDeviceToDevice));
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

__global__ void set_kernel(REAL* dst, REAL const value, std::size_t const count)
{
	std::size_t const index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    dst[index] = value;
}

void GPUData::set(REAL* arr, REAL const value, int const count)
{
    int const tx = 256;
	int const bx = (count / tx) + 1;

    dim3  threads(tx, 1, 1);
    dim3  blocks(bx, 1, 1);
    set_kernel<<< blocks, threads >>>(arr, value, count);
    CUDA_CHECK_STATUS(cudaGetLastError());
}

__global__ void cuda_point_to_data_sets(
    REAL ** pointer_to_pointers,
    REAL * pointer,
    std::size_t const n_pointers,
    std::size_t const size)
{
    std::size_t const index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n_pointers)
        return;

    int const begin = index * size;

    pointer_to_pointers[index] = pointer + begin;
}
#ifdef USE_CUBLAS

void GPUData::point_to_data_sets()
{
    dim3  threads(1, 1, 1);
    dim3  blocks(1, 1, 1);

    std::size_t max_threads = 256;

    threads.x
        = static_cast<unsigned int>
          (std::min(info_.max_chunk_size_, max_threads));
    blocks.x
        = static_cast<unsigned int>
          (std::ceil(REAL(info_.max_chunk_size_) / REAL(threads.x)));

    cuda_point_to_data_sets <<< blocks, threads >>>(
        pointer_decomposed_hessians_,
        decomposed_hessians_,
        info_.max_chunk_size_,
        info_.n_parameters_to_fit_*info_.n_parameters_to_fit_);

    cuda_point_to_data_sets <<< blocks, threads >>> (
        pointer_deltas_,
        deltas_,
        info_.max_chunk_size_,
        info_.n_parameters_to_fit_);
}

#endif // USE_CUBLAS