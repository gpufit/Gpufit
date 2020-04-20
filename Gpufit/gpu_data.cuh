#ifndef GPUFIT_GPU_DATA_CUH_INCLUDED
#define GPUFIT_GPU_DATA_CUH_INCLUDED

#include "info.h"
#ifdef USE_CUBLAS
#include "cublas_v2.h"
#endif

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <limits>

template< typename Type >
struct Device_Array
{
    explicit Device_Array(std::size_t const size) : allocated_size_(size)
    {
        std::size_t const maximum_size = std::numeric_limits< std::size_t >::max();
        std::size_t const type_size = sizeof(Type);
        if (size <= maximum_size / type_size)
        {
            cudaError_t const status = cudaMalloc(&data_, size * type_size);
            if (status == cudaSuccess)
            {
                return;
            }
            else
            {
                throw std::runtime_error(cudaGetErrorString(status));
            }
        }
        else
        {
            throw std::runtime_error("maximum array size exceeded");
        }
    }

    ~Device_Array() { if (allocated_size_ > 0) cudaFree(data_); }

    operator Type * () { return static_cast<Type *>(data_); }
    operator Type const * () const { return static_cast<Type *>(data_); }

    Type const * data() const
    {
        return static_cast<Type *>(data_);
    }

    void assign(Type const * data)
    {
        data_ = const_cast<Type *>(data);
    }

    Type * copy(std::size_t const size, Type * const to) const
    {
        // TODO check size parameter

        std::size_t const type_size = sizeof(Type);
        cudaError_t const status
            = cudaMemcpy(to, data_, size * type_size, cudaMemcpyDeviceToHost);
        if (status == cudaSuccess)
        {
            return to + size;
        }
        else
        {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    }

private:
    void * data_;
    std::size_t allocated_size_;
};

class GPUData
{
public:
    GPUData(Info const & info);
    ~GPUData();

    void init
    (
        int const chuk_size,
        int const chunk_index,
        REAL const * data,
        REAL const * weights,
        REAL const * initial_parameters,
        std::vector<int> const & parameters_to_fit_indices,
        REAL const * constraints,
        int const * constraint_types,
        int * states,
        REAL * chi_squares,
        int * n_iterations
    );
    void init_user_info(char const * user_info);

    void read(bool * dst, int const * src);
    void set(int* arr, int const value);
    void set(REAL* arr, REAL const value, int const count);
    void copy(REAL * dst, REAL const * src, std::size_t const count);

private:

    void set(int* arr, int const value, int const count);
    void write(REAL* dst, REAL const * src, int const count);
    void write(int * dst, int const * src, int const count);
    void write(int* dst, std::vector<int> const & src);
    void write(char* dst, char const * src, std::size_t const count);
    void point_to_data_sets();

private:
    int chunk_size_;
    Info const & info_;

public:
    int chunk_index_;

    cublasHandle_t cublas_handle_;

    Device_Array< REAL > data_;
    Device_Array< REAL > weights_;
    Device_Array< REAL > parameters_;
    Device_Array< REAL > prev_parameters_;
    Device_Array< int > parameters_to_fit_indices_;
    Device_Array< REAL > constraints_;
    Device_Array< int > constraint_types_;
    Device_Array< char > user_info_;

    Device_Array< REAL > chi_squares_;
    Device_Array< REAL > prev_chi_squares_;
    Device_Array< REAL > gradients_;
    Device_Array< REAL > hessians_;
    Device_Array< REAL > deltas_;
    Device_Array< REAL > scaling_vectors_;
    Device_Array< REAL > subtotals_;

    Device_Array< REAL > values_;
    Device_Array< REAL > derivatives_;

    Device_Array< REAL > lambdas_;
    Device_Array< int > states_;
    Device_Array< int > finished_;
    Device_Array< int > iteration_failed_;
    Device_Array< int > all_finished_;
    Device_Array< int > n_iterations_;
    Device_Array< int > solution_info_;

#ifdef USE_CUBLAS
    Device_Array< REAL > decomposed_hessians_;
    Device_Array< REAL * > pointer_decomposed_hessians_;
    Device_Array< REAL * > pointer_deltas_;
    Device_Array< int > pivot_vectors_;
#endif // USE_CUBLAS
};

#endif
