// HIP-only shim so the USE_CUBLAS path's `#include "cublas_v2.h"` resolves on
// ROCm (where no CUDA headers exist). This directory is added to the include
// path ONLY for the HIP build; the NVIDIA build uses the real cuBLAS header.
// Routed through cuda_to_hip.h, which (under USE_CUBLAS) includes hipBLAS and
// aliases the cublas* batched-LU symbols Gpufit uses to their hipblas*
// equivalents.
#pragma once
#include "../cuda_to_hip.h"
