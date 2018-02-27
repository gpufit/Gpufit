#ifndef GPUFIT_GPU_DATA_CUH_INCLUDED
#define GPUFIT_GPU_DATA_CUH_INCLUDED

#include "info.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"
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

    Type const * data() const
    {
        return static_cast<Type *>(data_);
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
        double const * data,
        double const * weights,
        double const * initial_parameters,
        std::vector<int> const & parameters_to_fit_indices
    );
    void init_user_info(char const * user_info);

    void read(bool * dst, int const * src);
    void set(int* arr, int const value);
    void set(int* arr, int const value, int const count);
    void set(double* arr, double const value, int const count);
    void copy(double * dst, double const * src, std::size_t const count);

private:

    void write(double* dst, double const * src, int const count);
    void write(int* dst, std::vector<int> const & src);
    void write(char* dst, char const * src, std::size_t const count);

private:
    int chunk_size_;
    Info const & info_;

public:
    int chunk_index_;

    cublasHandle_t cublas_handle_;

    Device_Array< double > data_;
    Device_Array< double > weights_;
    Device_Array< double > parameters_;
    Device_Array< double > prev_parameters_;
    Device_Array< int > parameters_to_fit_indices_;
    Device_Array< char > user_info_;

    Device_Array< double > chi_squares_;
    Device_Array< double > prev_chi_squares_;
    Device_Array< double > gradients_;
    Device_Array< double > hessians_;
    Device_Array< double > scaled_hessians_;
    Device_Array< double > deltas_;
    Device_Array< double > scaling_vectors_;

    Device_Array< double > values_;
    Device_Array< double > derivatives_;
    Device_Array< double > temp_derivatives_;

    Device_Array< double > lambdas_;
    Device_Array< double > lambda_lower_bounds_;
    Device_Array< double > lambda_upper_bounds_;
    Device_Array< double > step_bounds_;
    Device_Array< double > actual_reductions_;
    Device_Array< double > predicted_reductions_;
    Device_Array< double > directive_derivatives_;
    Device_Array< double > approximation_ratios_;
    Device_Array< double > scaled_parameters_;
    Device_Array< double > scaled_deltas_;
    Device_Array< double > scaled_delta_norms_;
    Device_Array< double > phis_;
    Device_Array< double > phi_derivatives_;

    Device_Array< int > states_;
    Device_Array< int > finished_;
    Device_Array< int > iteration_failed_;
    Device_Array< int > lambda_accepted_;
    Device_Array< int > newton_step_accepted_;
    Device_Array< int > all_finished_;
    Device_Array< int > all_lambdas_accepted_;
    Device_Array< int > n_iterations_;

    Device_Array< double > decomposed_hessians_;
    Device_Array< double > inverted_hessians_;
    Device_Array< double * > pointer_decomposed_hessians_;
    Device_Array< double * > pointer_inverted_hessians_;
    Device_Array< double * > pointer_deltas_;
    Device_Array< int > pivot_vectors_;
    Device_Array< int > cublas_info_;
};

#endif
