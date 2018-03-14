#include "info.h"
#include <cuda_runtime.h>

void Info::get_gpu_properties()
{
    cudaDeviceProp devProp;
    CUDA_CHECK_STATUS(cudaGetDeviceProperties(&devProp, 0));
    max_threads_ = devProp.maxThreadsPerBlock;
    max_blocks_ = devProp.maxGridSize[0];

    std::size_t free_bytes;
    std::size_t total_bytes;
    CUDA_CHECK_STATUS(cudaMemGetInfo(&free_bytes, &total_bytes));
    available_gpu_memory_ = std::size_t(double(free_bytes) * 0.1);
    
    if (double(user_info_size_) > double(free_bytes) * 0.9)
    {
        throw std::runtime_error("maximum user info size exceeded");
    }
}

int getDeviceCount()
{
	int deviceCount;
	CUDA_CHECK_STATUS(cudaGetDeviceCount(&deviceCount));
	return deviceCount;
}