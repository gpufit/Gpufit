// HIP-only shim so source that does `#include <device_launch_parameters.h>`
// resolves on ROCm. HIP provides threadIdx/blockIdx/blockDim/gridDim and the
// kernel launch builtins via <hip/hip_runtime.h>, pulled in by cuda_to_hip.h;
// there is no separate launch-parameters header on ROCm. HIP build only.
#pragma once
#include "../cuda_to_hip.h"
