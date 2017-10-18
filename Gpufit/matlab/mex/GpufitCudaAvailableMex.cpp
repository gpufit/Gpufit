#include "Gpufit/gpufit.h"

#include <mex.h>

void mexFunction(
    int          nlhs,
    mxArray      *plhs[],
    int          nrhs,
    mxArray const *prhs[])
{
	int available = gpufit_cuda_available();
	plhs[0] = mxCreateDoubleScalar(static_cast<double>(available));
}
