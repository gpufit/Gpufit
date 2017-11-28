#ifndef GPUFIT_GPU_DATA_CUH_INCLUDED
#define GPUFIT_GPU_DATA_CUH_INCLUDED

#include "info.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>
#include <limits>

template< typename Type >
struct Device_Array
{
    explicit Device_Array(std::size_t const size)
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

    ~Device_Array() { cudaFree(data_); }

    operator Type * () { return static_cast<Type *>(data_); }
    operator Type const * () const { return static_cast<Type *>(data_); }

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
};

class GPUData
{
public:
    GPUData(Info const & info);

    void init
    (
        int const chuk_size,
        int const chunk_index,
        float const * data,
        float const * weights,
        float const * initial_parameters,
        std::vector<int> const & parameters_to_fit_indices
    );
    void init_user_info(char const * user_info);

    void read(bool * dst, int const * src);
    void set(int* arr, int const value);
    void set(float* arr, float const value, int const count);
    void copy(float * dst, float const * src, std::size_t const count);

private:

    void set(int* arr, int const value, int const count);
    void write(float* dst, float const * src, int const count);
    void write(int* dst, std::vector<int> const & src);
    void write(char* dst, char const * src, std::size_t const count);

private:
    int chunk_size_;
    Info const & info_;

public:
    int chunk_index_;

    Device_Array< float > data_;
    Device_Array< float > weights_;
    Device_Array< float > parameters_;
    Device_Array< float > prev_parameters_;
    Device_Array< int > parameters_to_fit_indices_;
    Device_Array< char > user_info_;

    Device_Array< float > chi_squares_;
    Device_Array< float > prev_chi_squares_;
    Device_Array< float > gradients_;
    Device_Array< float > hessians_;
    Device_Array< float > deltas_;

    Device_Array< float > values_;
    Device_Array< float > derivatives_;

    Device_Array< float > lambdas_;
    Device_Array< int > states_;
    Device_Array< int > finished_;
    Device_Array< int > iteration_failed_;
    Device_Array< int > all_finished_;
    Device_Array< int > n_iterations_;
};

#endif
