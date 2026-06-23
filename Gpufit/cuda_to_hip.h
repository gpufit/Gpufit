//********************************************************//
// CUDA-to-HIP compatibility shim for Gpufit              //
//                                                        //
// Minimal-footprint port: every other source file keeps  //
// its plain CUDA spelling. On AMD this header includes    //
// the HIP runtime and #defines the CUDA symbols the       //
// project uses to their HIP equivalents. On NVIDIA it is  //
// a no-op that pulls in <cuda_runtime.h>.                 //
//                                                        //
// Symbol names follow PyTorch's authoritative hipify map: //
// torch/utils/hipify/cuda_to_hip_mappings.py             //
//                                                        //
// Scope: Gpufit's default solver (USE_CUBLAS=OFF) uses    //
// the self-contained Gauss-Jordan kernel and no GPU math  //
// library, so only the CUDA runtime API needs aliasing.   //
// There are no warp intrinsics, textures, surfaces, or    //
// constant memory in the project, hence none are mapped.  //
//********************************************************//

#ifndef GPUFIT_CUDA_TO_HIP_H
#define GPUFIT_CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// ---- Error handling ----
#define cudaError              hipError_t
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaGetErrorString     hipGetErrorString
#define cudaGetLastError       hipGetLastError
#define cudaDeviceSynchronize  hipDeviceSynchronize

// ---- Version queries ----
#define cudaDriverGetVersion   hipDriverGetVersion
#define cudaRuntimeGetVersion  hipRuntimeGetVersion

// ---- Device management ----
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaSetDevice            hipSetDevice
#define cudaDeviceProp           hipDeviceProp_t
#define cudaMemGetInfo           hipMemGetInfo

// ---- Linear memory ----
#define cudaMalloc  hipMalloc
#define cudaFree    hipFree
#define cudaMemcpy  hipMemcpy
#define cudaMemset  hipMemset

// ---- memcpy kinds ----
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice  hipMemcpyDeviceToDevice

// ---- cuBLAS -> hipBLAS (optional batched-LU solver, USE_CUBLAS=ON) ----
// The default Gauss-Jordan solver uses no GPU BLAS, so none of this is pulled
// in unless USE_CUBLAS is requested. hipBLAS mirrors the cuBLAS batched-LU API,
// so the cublas* calls map 1:1. One signature difference is handled at the call
// site: getrsBatched takes float* const A[] on hipBLAS vs const float* const A[]
// on cuBLAS (see lm_fit_cuda.cu).
#if defined(USE_CUBLAS)
#include <hipblas/hipblas.h>
#define cublasHandle_t       hipblasHandle_t
#define cublasStatus_t       hipblasStatus_t
#define cublasCreate         hipblasCreate
#define cublasDestroy        hipblasDestroy
#define cublasSgetrfBatched  hipblasSgetrfBatched
#define cublasDgetrfBatched  hipblasDgetrfBatched
#define cublasSgetrsBatched  hipblasSgetrsBatched
#define cublasDgetrsBatched  hipblasDgetrsBatched
#define CUBLAS_OP_N          HIPBLAS_OP_N
#endif // USE_CUBLAS

#else // NVIDIA / CUDA

#include <cuda_runtime.h>

#endif

#endif // GPUFIT_CUDA_TO_HIP_H
