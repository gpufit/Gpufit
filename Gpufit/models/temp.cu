#include <stdio.h>
// Macro for converting subscripts to linear index:
#define INDEX_VOL_TIME(i, t) i*${T} +t
#define INDEX_JAC_TIME(i, t, p) i*${T} *${D} +t*${D} +p
#define INDEX_PARAM(i, p) i*${D} +p
#define INDEX_MASK(x, y, z) x*${W} *${L} +y*${L} +z

/*******************************************************************************************************************************
                          MODEL FUNCTIONS DECLARATION (see the end of this file for the body of the functions)
*******************************************************************************************************************************/
__device__ void bicomp_3expIF(unsigned int idx, float *aux_par, float *inputfuns, float *IF, float *times, float *func, float *jac, float *dk);
__device__ void bicompartment_3expIF_3k(float *aux_par, float *inputfun, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
/*******************************************************************************************************************************
                                                       PET
*******************************************************************************************************************************/

// BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 3 EXP (like in Feng model #2)
__device__ void bicompartment_3expIF_3k(float *aux_par, float *inputfun, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
{
	// Obtain the linear index corresponding to the current thread:
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//*** Uncomment the lines below to use shared memory
	__shared__ float times[${T}];
	__shared__ float inputfuns[7];
	if (threadIdx.x < ${T}) {
		times[threadIdx.x] = time[threadIdx.x];
		if (threadIdx.x < 7)
			inputfuns[threadIdx.x] = inputfun[threadIdx.x];
	}
	__syncthreads();
	bicomp_3expIF(idx, aux_par, inputfuns, IF, times, func, jac, dk);

	// deactivate the jacobian for the 4th kinetic constant we don't want to update
	for (uint tt=0; tt<${T}; ++tt) {
		jac[INDEX_JAC_TIME(idx,tt,4)] = 0;
	}
}

// ANALYTIC FORMULATION OF A BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 3 EXP (like in Feng model #2)
__device__ void bicomp_3expIF(unsigned int idx, float *aux_par, float *inputfuns, float *IF, float *times, float *func, float *jac, float *dk)
{
	float delta0;
	float delta;
	float p[4];
	float Ahat[3];
	float Abar[3];
	float sum[${T}];
	float TAC[${T}];
	float Jb[${T}];
	float Jl[${T}];

	// Compute output of bicompartmental model and Jacobian using analytical expression.
	if (idx < ${N}) {
		p[1] = aux_par[INDEX_PARAM(idx,2)];
		p[3] = aux_par[INDEX_PARAM(idx,4)];
		p[0] = aux_par[INDEX_PARAM(idx,1)];
		p[2] = aux_par[INDEX_PARAM(idx,3)];
		Abar[0] = -inputfuns[2]-inputfuns[3];
		Abar[1] =  inputfuns[2];
		Abar[2] =  inputfuns[3];

		for (uint tt=0; tt<${T}; ++tt) { // reset the values of TAC and JAC for current voxel/thread
			func[INDEX_VOL_TIME(idx,tt)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,0)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,1)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,2)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,3)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,4)] = 0;
			TAC[tt] = 0;
		}

		for (uint ii=0; ii<=2; ii+=2) { //i = 1:2:4 % 2 compartiments
			delta0  = p[ii+1] + inputfuns[4];
			Ahat[0] = -inputfuns[2]-inputfuns[3]-(inputfuns[1]/delta0);
			Ahat[1] = inputfuns[2];
			Ahat[2] = inputfuns[3];

			for (uint tt=0; tt<${T}; ++tt) { // reset temporary variables for i-th compartment
				sum[tt]=0;
				Jb[tt] =0;
				Jl[tt] =0;
			}
			for (uint jj=0; jj<3; ++jj) {
				delta  = p[ii+1]+inputfuns[4+jj];

				for (uint tt=0; tt<${T}; ++tt) {
					if (times[tt]>=inputfuns[0]) {
						sum[tt] += Ahat[jj] * (1.0f / delta) * ( exp(inputfuns[4+jj]*(times[tt]-inputfuns[0]))-exp(-p[ii+1]*(times[tt]-inputfuns[0])) );
						Jb[tt]  += Ahat[jj] * (1.0f / delta) * ( exp(inputfuns[4+jj]*(times[tt]-inputfuns[0]))-exp(-p[ii+1]*(times[tt]-inputfuns[0])) );
						Jl[tt]  += Abar[jj] * (1.0f / (delta*delta)) * ( exp(-p[ii+1]*(times[tt]-inputfuns[0]))-exp(inputfuns[4+jj]*(times[tt]-inputfuns[0]))) + Abar[jj] * (1.0f / delta)*(times[tt]-inputfuns[0]) * exp(-p[ii+1]*(times[tt]-inputfuns[0]));
					}
				}
			}
			for (uint tt=0; tt<${T}; ++tt) {
				if (times[tt]>=inputfuns[0]) {
					TAC[tt] += p[ii] * (sum[tt] + ((inputfuns[1]*(times[tt]-inputfuns[0]))/delta0) *exp(inputfuns[4]*(times[tt]-inputfuns[0])));
					jac[INDEX_JAC_TIME(idx,tt,ii+1)] = (1-aux_par[INDEX_PARAM(idx,0)]) * (Jb[tt] + ((inputfuns[1]*(times[tt]-inputfuns[0]))/delta0) *exp(inputfuns[4]*(times[tt]-inputfuns[0])));
					jac[INDEX_JAC_TIME(idx,tt,ii+2)] = (1-aux_par[INDEX_PARAM(idx,0)]) * (p[ii] * (Jl[tt] + ( exp(-p[ii+1]*(times[tt]-inputfuns[0]))-exp(inputfuns[4]*(times[tt]-inputfuns[0]))) * (inputfuns[1] *(times[tt]-inputfuns[0]) * (1.0f / (delta0*delta0)) + 2*inputfuns[1] * (1.0f / (delta0*delta0*delta0))) ));
				}
			}

		}
		for (uint tt=0; tt<${T}; ++tt) {
			TAC[tt]  *= exp(-dk[0]*times[tt]);
			jac[INDEX_JAC_TIME(idx,tt,0)] = IF[tt] - TAC[tt];
			TAC[tt]  = ((1-aux_par[INDEX_PARAM(idx,0)]) * TAC[tt]) + (aux_par[INDEX_PARAM(idx,0)] * IF[tt]);
			if (TAC[tt] < 0.0) {
				TAC[tt] = 1e-16;
			}
			func[INDEX_VOL_TIME(idx,tt)] += TAC[tt];
		}
	}
}
